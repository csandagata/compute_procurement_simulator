// Procurement simulation core (Tier 3 refactor: workload portfolio).
//
// What's new vs. v2:
//   - Plans now consume a *portfolio* of workload-class instances rather than
//     hard-coded LLM token volumes. The buyer composes a mix from a library
//     spanning LLM, biology, physics, climate, earth-obs, industrial-twin,
//     and defense/intel domains; each item has its own growth model and
//     scaling-case (low / median / high / fitted).
//   - Per-workload-class throughput dispatch: autoregressive / single-pass /
//     iterative-simulation / active-learning-loop / continuous-training each
//     get the right physics formula and the right SKU eligibility filter
//     (FP64-required workloads exclude B200; CUDA-only workloads exclude AMD).
//   - Latent / unfitted scaling laws are first-class: low/median/high cases
//     let the buyer ask "if AlphaFold turns out to be compute-bound and scales
//     2x/yr, what's our fleet?" alongside "if it's data-bound and stays flat".
//
// Backward compatibility: if `inputs.portfolio` is missing, a portfolio is
// synthesized from the legacy {interactiveTokensPerDay, ...} inputs so the
// existing scenarios and charts keep working unchanged.

import { GPUS, RENTAL_PRICE_USD_PER_GPU_HOUR, NETWORK_COST_PER_GPU_USD, DC } from "./gpus.js";
import { kvBytesPerToken as kvBytesFromScheme } from "./kv.js";
import { specDecSpeedup } from "./spec_dec.js";
import { utilizationCeiling, p99WaitMs } from "./queueing.js";
import { placeMwAcrossRegions, DEFAULT_REGIONS } from "./sites.js";
import { optimizeServingConfig } from "./throughput.js";
import { WORKLOAD_CLASSES, legacyCategoryOf } from "./workload_classes.js";
import { planWorkload } from "./workload_throughput.js";
import {
  makePortfolioItem,
  resolveWorkload,
  demandAtYear,
} from "./portfolio.js";

const HOURS_PER_YEAR = 24 * 365;

function bytesPerParam(quantization) {
  switch (quantization) {
    case "fp16": return 2;
    case "fp8":  return 1;
    case "int4": return 0.5;
    default:     return 2;
  }
}

// ---------------------------------------------------------------------------
// Year scaling helpers
// ---------------------------------------------------------------------------
export function modelGrowthFactor(rate, year)   { return Math.pow(1 + rate, year); }
export function demandGrowthFactor(rate, year)  { return Math.pow(1 + rate, year); }
export function efficiencyMultiplier(rate, year){ return Math.pow(1 + rate, year); }
export function externalPricePerMtok(initialUsd, declineMultPerYear, year) {
  return initialUsd / Math.pow(declineMultPerYear, year);
}

// Smallest pod size that can hold the model weights at this quantization.
export function minGpusForModel({ paramsB, quantization, gpu, kvHeadroomGb = 30 }) {
  const wpBytes = bytesPerParam(quantization);
  const weightGb = (paramsB * 1e9 * wpBytes) / 1e9;
  const usableHbm = gpu.hbm_gb * 0.85 - kvHeadroomGb;
  if (usableHbm <= 0) return Infinity;
  return Math.max(1, Math.ceil(weightGb / usableHbm));
}

// Resolve KV bytes/token: explicit override wins; otherwise derive from scheme.
function resolveKvBytes(inputs, paramsBYear) {
  if (inputs.kvSchemeOverride && inputs.kvBytesPerToken && inputs.kvBytesPerToken > 0) {
    return inputs.kvBytesPerToken;
  }
  return kvBytesFromScheme({
    scheme: inputs.kvScheme || "GQA_8",
    paramsB: paramsBYear,
    mlaCompressionDim: inputs.mlaCompressionDim || 576,
  });
}

// ---------------------------------------------------------------------------
// Legacy → portfolio synthesis (backwards compatibility)
// ---------------------------------------------------------------------------

// Synthesize a portfolio from the old token-volume / FLOPs-budget inputs.
// Tokens/year = tokens/day * 365. Growth rates are encoded as customYoy on
// each item so demandAtYear() returns the same series the old buildPlan did.
function synthesizePortfolioFromLegacy(inputs) {
  const items = [];
  const dRate = inputs.demandGrowth ?? 0;
  const mRate = inputs.modelGrowth ?? 0;
  // Pretrain effective rate: (1+m)^2 * (1+d) - 1 (Chinchilla mGrow²·dGrow)
  const pretrainEff = Math.pow(1 + mRate, 2) * (1 + dRate) - 1;
  const finetuneEff = (1 + mRate) * (1 + dRate) - 1;

  if ((inputs.interactiveTokensPerDay ?? 0) > 0) {
    items.push(makePortfolioItem("llm_interactive", {
      unitsPerYear: inputs.interactiveTokensPerDay * 365,
      growthModel: "exponential",
    }));
    items[items.length - 1].customYoy = dRate;
  }
  if ((inputs.batchTokensPerDay ?? 0) > 0) {
    items.push(makePortfolioItem("llm_batch", {
      unitsPerYear: inputs.batchTokensPerDay * 365,
      growthModel: "exponential",
    }));
    items[items.length - 1].customYoy = dRate;
  }
  if ((inputs.rlTokensPerDay ?? 0) > 0) {
    items.push(makePortfolioItem("llm_rl_rollout", {
      unitsPerYear: inputs.rlTokensPerDay * 365,
      growthModel: "exponential",
    }));
    items[items.length - 1].customYoy = dRate;
  }
  if ((inputs.pretrainFlopsPerYear ?? 0) > 0) {
    items.push(makePortfolioItem("llm_pretrain", {
      unitsPerYear: inputs.pretrainFlopsPerYear,
      growthModel: "exponential",
    }));
    items[items.length - 1].customYoy = pretrainEff;
  }
  if ((inputs.finetuneFlopsPerYear ?? 0) > 0) {
    items.push(makePortfolioItem("llm_finetune", {
      unitsPerYear: inputs.finetuneFlopsPerYear,
      growthModel: "exponential",
    }));
    items[items.length - 1].customYoy = finetuneEff;
  }
  return items;
}

function getPortfolio(inputs) {
  if (Array.isArray(inputs.portfolio) && inputs.portfolio.length > 0) {
    return inputs.portfolio;
  }
  return synthesizePortfolioFromLegacy(inputs);
}

// ---------------------------------------------------------------------------
// Per-year planning
// ---------------------------------------------------------------------------

function autoregressiveLatencyTargets(workload) {
  const tier = workload.latency_tier || "regional";
  const decodeMs = { edge: 30, regional: 100, central: 500, batch: 1000 }[tier] ?? 100;
  const prefillMs = { edge: 600, regional: 5000, central: 30000, batch: 60000 }[tier] ?? 5000;
  return { decodeMs, prefillMs };
}

function buildYearPlan(year, inputs, portfolio) {
  const horizonYears = inputs.horizonYears ?? 5;
  const allowedSkus = (inputs.allowedGpus && inputs.allowedGpus.length > 0)
    ? inputs.allowedGpus
    : Object.keys(GPUS);
  const electricityPrice = inputs.electricityPrice ?? 0.07;
  const pue = inputs.pue ?? 1.25;
  const utilization = inputs.utilization ?? 0.6;
  const mfu = inputs.mfu ?? 0.35;

  const mGrow = modelGrowthFactor(inputs.modelGrowth ?? 0, year);
  const algMul = efficiencyMultiplier(inputs.algEfficiency ?? 0, year);
  const paramsBYear = (inputs.paramsB ?? 70) * mGrow;
  const kvBytesPerTok = resolveKvBytes(inputs, paramsBYear);

  const ctx = {
    year, paramsBYear, kvBytesPerTok,
    activeFrac: inputs.activeFrac ?? 1.0,
    quantization: inputs.quantization ?? "fp8",
    contextLen: inputs.contextLen ?? 4000,
    outputLen: inputs.outputLen ?? 200,
    electricityPrice, pue, utilization, mfu,
    maxTp: inputs.maxTpPerWorkload || null,
    enableQueueing: inputs.enableQueueing !== false,
  };

  // Plan each portfolio item.
  const itemPlans = {};
  for (let i = 0; i < portfolio.length; i++) {
    const item = portfolio[i];
    if (!item.enabled) continue;
    const workload = resolveWorkload(item);
    if (!workload) continue;
    const demand = demandAtYear(item, year);
    if (demand <= 0) {
      itemPlans[item.classId] = { item, workload, demand: 0, plan: { gpuYears: 0, sku: null, pattern: workload.compute_pattern } };
      continue;
    }
    // Configure spec-dec multiplier per workload (only autoregressive items benefit).
    const useSpec = inputs.enableSpecDec && workload.benefits_from_spec_dec;
    const specMul = useSpec
      ? specDecSpeedup({
          acceptanceProb: inputs.specDecAcceptanceProb,
          gammaMax: inputs.specDecGammaMax,
          draftToTargetCostRatio: inputs.specDecDraftCost ?? 0.08,
        })
      : 1.0;
    const lat = workload.compute_pattern === "autoregressive"
      ? autoregressiveLatencyTargets(workload)
      : { decodeMs: Infinity, prefillMs: Infinity };
    const plan = planWorkload({
      workload, allowedSkus, demandUnitsPerYear: demand,
      ctx: { ...ctx, specMul, decodeLatencyMs: lat.decodeMs, prefillLatencyMs: lat.prefillMs },
      algMul,
    });
    itemPlans[item.classId] = { item, workload, demand, plan, specMul };
  }

  // Aggregate per SKU.
  const gpusBySku = {};
  const addSku = (sku, gy) => {
    if (!sku || !isFinite(gy) || gy <= 0) return;
    gpusBySku[sku] = (gpusBySku[sku] || 0) + gy;
  };
  for (const ip of Object.values(itemPlans)) {
    const p = ip.plan;
    if (!p) continue;
    if (p.pattern === "autoregressive") {
      addSku(p.prefill?.sku, p.prefill?.gpuYears);
      addSku(p.decode?.sku, p.decode?.gpuYears);
    } else {
      addSku(p.sku, p.gpuYears);
    }
  }

  // Bucket into legacy categories (for existing charts).
  const gpuYearsByLegacy = {
    interactive_inference: 0, batch_inference: 0, rl: 0, training: 0, finetune: 0,
  };
  // And per-domain (NEW: powers domain-color charts later).
  const gpuYearsByDomain = {};
  for (const [classId, ip] of Object.entries(itemPlans)) {
    const cls = WORKLOAD_CLASSES[classId];
    if (!cls) continue;
    const gy = isFinite(ip.plan?.gpuYears) ? ip.plan.gpuYears : 0;
    const cat = legacyCategoryOf(classId);
    gpuYearsByLegacy[cat] = (gpuYearsByLegacy[cat] || 0) + gy;
    gpuYearsByDomain[cls.domain] = (gpuYearsByDomain[cls.domain] || 0) + gy;
  }

  const totalGpuYears = Object.values(gpuYearsByLegacy)
    .reduce((a, b) => a + (isFinite(b) ? b : 0), 0);

  const headroom = inputs.headroomFactor ?? 1.25;
  const totalGpus = Math.ceil(totalGpuYears * headroom);

  const rentalShare = inputs.rentalShare ?? 0.0;
  const ownedShare = 1 - rentalShare;
  const ownedGpus = Math.ceil(totalGpus * ownedShare);
  const rentedGpus = totalGpus - ownedGpus;

  // Pick most-used SKU as headline for legacy chart code.
  let headlineSku = null, headlineCount = 0;
  for (const [sku, count] of Object.entries(gpusBySku)) {
    if (count > headlineCount) { headlineSku = sku; headlineCount = count; }
  }
  if (!headlineSku) headlineSku = allowedSkus[0];
  const headlineGpu = GPUS[headlineSku];

  const minScaleUpGpus = minGpusForModel({
    paramsB: paramsBYear,
    quantization: inputs.quantization,
    gpu: headlineGpu,
    kvHeadroomGb: 30,
  });

  // Power: weighted across SKU mix (scaled by headroom).
  let totalKw = 0;
  for (const [sku, count] of Object.entries(gpusBySku)) {
    const gpu = GPUS[sku];
    if (!gpu) continue;
    totalKw += (count * headroom) * (gpu.power_w / 1000) * pue;
  }

  // Site placement.
  const sitePlacement = placeMwAcrossRegions({
    mwNeededByYear: [totalKw / 1000],
    regions: inputs.regions || DEFAULT_REGIONS,
    preferredOrder: inputs.regionPreferredOrder,
    rentalShareOverride: rentalShare,
  });
  const siteThisYear = sitePlacement[0];

  // Latency-tier facility split (legacy chart compat).
  const ownedTierGpus = {
    edge:     Math.ceil(ownedGpus * (inputs.edgeFacilityShare     ?? 0.2)),
    regional: Math.ceil(ownedGpus * (inputs.regionalFacilityShare ?? 0.4)),
    central:  Math.ceil(ownedGpus * (inputs.centralFacilityShare  ?? 0.4)),
  };
  const tierFacilityCount = {};
  for (const [tier, count] of Object.entries(ownedTierGpus)) {
    const mwPerGpu = ((headlineGpu?.power_w ?? 700) / 1000) * pue / 1000;
    const mwTotal = count * mwPerGpu;
    tierFacilityCount[tier] = Math.max(count > 0 ? 1 : 0,
      Math.ceil(mwTotal / (inputs.facilityMaxMw ?? 30)));
  }
  const numFacilities = Object.values(tierFacilityCount).reduce((a, b) => a + b, 0);

  // CapEx / OpEx (weighted by SKU mix).
  let weightedCapex = 0, weightedNetwork = 0, weightedRental = 0, totalShare = 0;
  for (const [sku, count] of Object.entries(gpusBySku)) {
    const gpu = GPUS[sku];
    if (!gpu) continue;
    weightedCapex   += count * gpu.capex_usd;
    weightedNetwork += count * (NETWORK_COST_PER_GPU_USD[inputs.networkTier ?? "small_pod"] ?? 8000);
    weightedRental  += count * (RENTAL_PRICE_USD_PER_GPU_HOUR[sku] ?? 2.5);
    totalShare      += count;
  }
  const avgGpuCapex   = totalShare > 0 ? weightedCapex / totalShare : (headlineGpu?.capex_usd ?? 28000);
  const avgNetworkCap = totalShare > 0 ? weightedNetwork / totalShare : 8000;
  const avgRentalRate = totalShare > 0 ? weightedRental / totalShare : 2.0;

  return {
    year, paramsBYear, kvBytesPerTok,
    itemPlans,                                          // NEW
    gpusBySku,                                          // already there
    gpuYears: gpuYearsByLegacy,                         // legacy chart shape
    gpuYearsByDomain,                                   // NEW
    totalGpuYears, totalGpus, ownedGpus, rentedGpus,
    headlineSku, headlineGpu,
    minScaleUpGpus,
    totalKw, annualKwh: totalKw * HOURS_PER_YEAR,
    sitePlacement: siteThisYear, siteShortfall: siteThisYear?.shortfall ?? 0,
    ownedTierGpus, tierFacilityCount, numFacilities,
    avgGpuCapex, avgNetworkCap, avgRentalRate,
  };
}

// ---------------------------------------------------------------------------
// Main entry: buildPlan
// ---------------------------------------------------------------------------

export function buildPlan(inputs) {
  const portfolio = getPortfolio(inputs);
  const horizonYears = inputs.horizonYears ?? 5;
  const refreshYears = inputs.refreshYears ?? 4;
  const electricityPrice = inputs.electricityPrice ?? 0.07;

  const yearly = [];
  for (let y = 0; y < horizonYears; y++) {
    const yp = buildYearPlan(y, inputs, portfolio);
    yearly.push(yp);
  }

  // Year-over-year financials (need to look across years for incremental CapEx).
  for (let y = 0; y < yearly.length; y++) {
    const yp = yearly[y];
    const prev = y > 0 ? yearly[y - 1] : null;
    const isRefreshYear = (y === 0) || (y > 0 && y % refreshYears === 0);

    const incrementalGpus = y === 0
      ? yp.ownedGpus
      : Math.max(0, yp.ownedGpus - (prev?.ownedGpus ?? 0))
        + (isRefreshYear ? (yearly[y - refreshYears]?.ownedGpus ?? 0) : 0);

    const gpuCapex = incrementalGpus * yp.avgGpuCapex;
    const networkCapex = incrementalGpus * yp.avgNetworkCap;
    const facilityCapex = isRefreshYear ? yp.totalKw * DC.facility_capex_per_kw_usd : 0;
    const facilityOpex = yp.totalKw * DC.facility_capex_per_kw_usd * DC.facility_opex_pct_of_capex;
    const electricityUsd = yp.annualKwh * electricityPrice;
    const rentalUsd = yp.rentedGpus * yp.avgRentalRate * HOURS_PER_YEAR;

    const totalCapex = gpuCapex + networkCapex + facilityCapex;
    const totalOpex  = electricityUsd + facilityOpex + rentalUsd;
    const yearTco    = totalCapex + totalOpex;
    const discounted = yearTco / Math.pow(1 + (inputs.discountRate ?? 0.10), y);

    // Internal $/Mtok: allocate TCO to LLM-inference workloads only.
    const inferenceGpuYears = (yp.gpuYears.interactive_inference || 0)
                            + (yp.gpuYears.batch_inference || 0);
    const inferenceShare = yp.totalGpuYears > 0 ? inferenceGpuYears / yp.totalGpuYears : 0;
    let tokensServed = 0;
    for (const ip of Object.values(yp.itemPlans)) {
      const cls = WORKLOAD_CLASSES[ip.item.classId];
      if (!cls) continue;
      if (cls.unit === "token" && cls.legacy_category !== "rl" && cls.legacy_category !== "training") {
        tokensServed += ip.demand;
      }
    }
    const inferenceTco = yearTco * inferenceShare;
    const internalDollarsPerMtok = tokensServed > 0
      ? (inferenceTco * 1e6) / tokensServed : 0;
    const externalDollarsPerMtok = externalPricePerMtok(
      inputs.externalInitialPrice ?? 5.0,
      inputs.externalDeclineMult ?? 3.0,
      y,
    );

    Object.assign(yp, {
      capex: { gpu: gpuCapex, network: networkCapex, facility: facilityCapex },
      opex:  { electricity: electricityUsd, facility: facilityOpex, rental: rentalUsd },
      yearTco, discounted,
      internalDollarsPerMtok, externalDollarsPerMtok,
      tokensServed,
    });

    // Backwards-compat aliases used by ui.js narrative.
    const interactivePlan = Object.values(yp.itemPlans)
      .find((ip) => ip.item.classId === "llm_interactive")?.plan;
    if (interactivePlan && interactivePlan.pattern === "autoregressive") {
      yp.allocations = yp.allocations || {};
      yp.allocations.interactive_inference = {
        prefill: { ...interactivePlan.prefill, infeasible: !interactivePlan.prefill?.sku },
        decode:  { ...interactivePlan.decode,  infeasible: !interactivePlan.decode?.sku,
                   p99WaitMs: interactivePlan.decode?.p99WaitMs ?? null,
                   utilization: yp.utilization ?? (inputs.utilization ?? 0.6) },
        specMul: Object.values(yp.itemPlans).find((ip) => ip.item.classId === "llm_interactive")?.specMul ?? 1.0,
      };
    } else {
      yp.allocations = { interactive_inference: { prefill: {}, decode: {} } };
    }
  }

  const totalDiscountedTco = yearly.reduce((a, b) => a + (isFinite(b.discounted) ? b.discounted : 0), 0);
  const peakGpus = Math.max(...yearly.map((y) => y.totalGpus));
  const peakMw = Math.max(...yearly.map((y) => y.totalKw / 1000));
  const peakFacilities = Math.max(...yearly.map((y) => y.numFacilities));
  const headlineSku = yearly[0]?.headlineSku ?? Object.keys(GPUS)[0];
  const gpu = GPUS[headlineSku];
  const anyShortfall = yearly.some((y) => y.siteShortfall > 0);

  return {
    yearly,
    portfolio,
    totalDiscountedTco,
    peakGpus, peakMw, peakFacilities,
    gpu, headlineSku,
    anyShortfall,
  };
}

// ---------------------------------------------------------------------------
// Sensitivity & uncertainty
// ---------------------------------------------------------------------------

export function tornado(baseInputs, lever, lo, hi) {
  const loInputs = { ...baseInputs, [lever]: lo };
  const hiInputs = { ...baseInputs, [lever]: hi };
  const loPlan = buildPlan(loInputs);
  const hiPlan = buildPlan(hiInputs);
  return { lever, lo: loPlan.totalDiscountedTco, hi: hiPlan.totalDiscountedTco };
}

export function tornadoBattery(baseInputs, baseTco) {
  const levers = [
    ["interactiveTokensPerDay", (baseInputs.interactiveTokensPerDay ?? 1e9) * 0.5,
                                (baseInputs.interactiveTokensPerDay ?? 1e9) * 2.0],
    ["batchTokensPerDay",       (baseInputs.batchTokensPerDay ?? 1e9) * 0.5,
                                (baseInputs.batchTokensPerDay ?? 1e9) * 2.0],
    ["demandGrowth",            (baseInputs.demandGrowth ?? 0.5) - 0.15,
                                (baseInputs.demandGrowth ?? 0.5) + 0.15],
    ["modelGrowth",             (baseInputs.modelGrowth ?? 0.4) - 0.10,
                                (baseInputs.modelGrowth ?? 0.4) + 0.10],
    ["algEfficiency",           (baseInputs.algEfficiency ?? 0.5) - 0.20,
                                (baseInputs.algEfficiency ?? 0.5) + 0.20],
    ["mfu",                     Math.max(0.05, (baseInputs.mfu ?? 0.35) - 0.15),
                                Math.min(0.7,  (baseInputs.mfu ?? 0.35) + 0.15)],
    ["utilization",             Math.max(0.2,  (baseInputs.utilization ?? 0.6) - 0.2),
                                Math.min(0.95, (baseInputs.utilization ?? 0.6) + 0.2)],
    ["pretrainFlopsPerYear",    (baseInputs.pretrainFlopsPerYear ?? 1e23) * 0.5,
                                (baseInputs.pretrainFlopsPerYear ?? 1e23) * 2],
    ["electricityPrice",        (baseInputs.electricityPrice ?? 0.07) * 0.5,
                                (baseInputs.electricityPrice ?? 0.07) * 1.5],
    ["rentalShare",             Math.max(0, (baseInputs.rentalShare ?? 0) - 0.25),
                                Math.min(1, (baseInputs.rentalShare ?? 0) + 0.25)],
  ];
  return levers.map(([k, lo, hi]) => {
    const r = tornado(baseInputs, k, lo, hi);
    return { lever: k, lo: r.lo - baseTco, hi: r.hi - baseTco, loVal: lo, hiVal: hi };
  });
}

export function demandFan(baseInputs, sigma = 0.4) {
  const out = { years: [], p10: [], p50: [], p90: [], capacity: [] };
  const base = buildPlan(baseInputs);
  for (let i = 0; i < base.yearly.length; i++) {
    const y = base.yearly[i];
    const median = y.totalGpuYears;
    const z = sigma * (i + 1) / Math.sqrt(base.yearly.length);
    out.years.push(y.year);
    out.p50.push(median);
    out.p10.push(median * Math.exp(-1.28 * z));
    out.p90.push(median * Math.exp(1.28 * z));
    out.capacity.push(y.totalGpus);
  }
  return out;
}

// Sweep portfolio scaling cases (low / median / high) per workload-class
// instance and return a fan of total-fleet outcomes. Useful for visualizing
// the *latent scaling-law uncertainty* in non-LLM domains.
export function scalingCaseFan(baseInputs) {
  const cases = ["low", "median", "high"];
  const out = { years: [], low: [], median: [], high: [] };
  for (const c of cases) {
    const portfolio = (getPortfolio(baseInputs)).map((it) => ({ ...it, scalingCase: c }));
    const inputs = { ...baseInputs, portfolio };
    const plan = buildPlan(inputs);
    out[c] = plan.yearly.map((y) => y.totalGpus);
    if (out.years.length === 0) out.years = plan.yearly.map((y) => y.year);
  }
  return out;
}

export function latencyCostFrontier(inputs, year = 0) {
  const gpu = GPUS[inputs.gpuKey ?? Object.keys(GPUS)[0]];
  const yearParamsB = (inputs.paramsB ?? 70) * Math.pow(1 + (inputs.modelGrowth ?? 0), year);
  const kvBytesPerTok = resolveKvBytes(inputs, yearParamsB);
  const inputLen = inputs.contextLen ?? 4000;
  const points = [];
  const latencyTargets = [10, 20, 30, 50, 75, 100, 150, 250, 500, 1000];
  const tps = [1, 2, 4, 8, 16, 32, 64];
  const specMul = inputs.enableSpecDec
    ? specDecSpeedup({
        acceptanceProb: inputs.specDecAcceptanceProb,
        gammaMax: inputs.specDecGammaMax,
      })
    : 1.0;
  for (const lat of latencyTargets) {
    let bestPrice = Infinity;
    let bestTps = 0;
    let bestPoint = null;
    for (const tp of tps) {
      const cfg = optimizeServingConfig({
        phase: "decode",
        paramsB: yearParamsB,
        activeFrac: inputs.activeFrac ?? 1.0,
        quantization: inputs.quantization ?? "fp16",
        gpu,
        kvBytesPerToken: kvBytesPerTok,
        contextLen: inputLen, inputLen,
        latencyMsTarget: lat,
        electricityPrice: inputs.electricityPrice ?? 0.07,
        pue: inputs.pue ?? 1.25,
        specDecMultiplier: specMul,
        maxTp: tp,
      });
      if (!cfg) continue;
      if (cfg.dollarsPerMtok < bestPrice) {
        bestPrice = cfg.dollarsPerMtok;
        bestTps = cfg.tpsPerGpu;
        bestPoint = { tp: cfg.tp, batch: cfg.batch, latencyMs: cfg.latencyMs, bound: cfg.bound };
      }
    }
    if (isFinite(bestPrice)) {
      points.push({ targetLatencyMs: lat, dollarsPerMtok: bestPrice, tpsPerGpu: bestTps, ...bestPoint });
    }
  }
  return points;
}

export const _internal = { bytesPerParam, synthesizePortfolioFromLegacy, getPortfolio };
