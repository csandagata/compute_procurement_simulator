// Procurement simulation core (Tier 1+2 refactor).
//
// What's new vs. v1:
//   1. Disaggregated prefill/decode — every inference workload is split into
//      a prefill phase and a decode phase, each optimized independently.
//   2. Heterogeneous fleet — for each workload×phase×year we search across
//      the allowed GPU SKUs and pick the cheapest one that meets latency.
//   3. Site & power capacity — MW demand is placed across regions via a
//      supply curve; shortfall is reported when capacity is tight.
//   4. Per-workload (TP, PP, batch) optimization — the throughput optimizer
//      sweeps the full grid for each phase, not just one global TP.
//   5. Speculative decoding — decode-phase throughput is multiplied by the
//      acceptance-prob × gamma speedup model.
//   6. Queueing penalty — utilization ceiling is sqrt-c-derived and respects
//      the workload's p99 latency budget.
//   7. KV-cache schemes — kvBytesPerToken is derived from MHA/GQA/MLA + arch.
//
// Outputs are pure functions; the dashboard re-renders on every input change.

import { GPUS, RENTAL_PRICE_USD_PER_GPU_HOUR, NETWORK_COST_PER_GPU_USD, DC } from "./gpus.js";
import {
  LATENCY_TARGET_MS_PER_TOK,
  TTFT_TARGET_MS,
  BENEFITS_FROM_SPEC_DEC,
} from "./workloads.js";
import { kvBytesPerToken as kvBytesFromScheme, approximateArchitecture } from "./kv.js";
import { specDecSpeedup } from "./spec_dec.js";
import { utilizationCeiling, p99WaitMs } from "./queueing.js";
import { placeMwAcrossRegions, DEFAULT_REGIONS } from "./sites.js";
import {
  optimizeServingConfig,
  decodeThroughputPerGpu,
  prefillThroughputPerGpu,
} from "./throughput.js";

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

// ---------------------------------------------------------------------------
// Core: build per-year plan
// ---------------------------------------------------------------------------

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

// For an inference workload, run the optimizer across allowed GPUs for both
// prefill and decode phases. Returns the per-phase chosen configuration.
function planInferenceWorkload({
  inputs, year, paramsBYear, kvBytesPerTok,
  workloadKey, latencyMsTarget, ttftTargetMs,
  inputLen, contextLen, outputLen,
  tokensPerYear, electricityPrice, pue,
  algMul,
}) {
  const allowedSkus = (inputs.allowedGpus && inputs.allowedGpus.length > 0)
    ? inputs.allowedGpus
    : Object.keys(GPUS);
  const specMul = (inputs.enableSpecDec && BENEFITS_FROM_SPEC_DEC[workloadKey])
    ? specDecSpeedup({
        acceptanceProb: inputs.specDecAcceptanceProb,
        gammaMax: inputs.specDecGammaMax,
        draftToTargetCostRatio: inputs.specDecDraftCost ?? 0.08,
      })
    : 1.0;

  // Token-volume split between prefill and decode, by token *count*.
  const prefillTokensPerYear = tokensPerYear * (inputLen / (inputLen + outputLen));
  const decodeTokensPerYear  = tokensPerYear * (outputLen / (inputLen + outputLen));

  function chooseSku(phase, latencyTarget, latencyMode) {
    let best = null;
    for (const sku of allowedSkus) {
      const gpu = GPUS[sku];
      if (!gpu) continue;
      const cfg = optimizeServingConfig({
        phase,
        paramsB: paramsBYear,
        activeFrac: inputs.activeFrac,
        quantization: inputs.quantization,
        gpu,
        kvBytesPerToken: kvBytesPerTok,
        contextLen, inputLen,
        latencyMsTarget: latencyTarget,
        electricityPrice, pue,
        specDecMultiplier: phase === "decode" ? specMul : 1.0,
        maxTp: inputs.maxTpPerWorkload || (gpu.scaleup_domain || 8) * 4,
      });
      if (!cfg) continue;
      if (!best || cfg.dollarsPerMtok < best.config.dollarsPerMtok) {
        best = { sku, gpu, config: cfg };
      }
    }
    return best;
  }

  const decodeChoice  = chooseSku("decode",  latencyMsTarget, "decode");
  // Prefill latency target is total prefill time (not per-token), so we feed
  // ttftTargetMs directly; the optimizer treats it as a total-time budget.
  const prefillChoice = chooseSku("prefill", ttftTargetMs ?? Infinity, "prefill");

  // Apply queueing-derived utilization ceiling (interactive only).
  // For batch workloads we use a fixed-high utilization since they're throughput
  // jobs and don't have p99-style latency contracts.
  const isInteractive = (workloadKey === "interactive_inference" || workloadKey === "rl");
  const baseUtil = inputs.utilization ?? 0.6;

  function gpuYearsForPhase(choice, tokensThisPhase, isPrefill) {
    if (!choice) return { gpuYears: Infinity, infeasible: true };
    const { config } = choice;
    if (!config || config.tpsPerGpu <= 0) return { gpuYears: Infinity, infeasible: true };
    const tpsPerGpu = config.tpsPerGpu * algMul;
    // Pre-pass utilization estimate: assume cServers ≈ tokens/sec / per-gpu-tps
    const peakReqRateTpsPerGpuFleet = tokensThisPhase / (HOURS_PER_YEAR * 3600);
    const cServersInitial = Math.max(1, peakReqRateTpsPerGpuFleet / tpsPerGpu);
    const utilCap = inputs.enableQueueing && isInteractive
      ? utilizationCeiling({ cServers: cServersInitial })
      : 0.85;
    const effectiveUtil = Math.min(baseUtil, utilCap);
    const tokPerGpuYear = tpsPerGpu * 3600 * 24 * 365 * effectiveUtil;
    const gpuYears = tokensThisPhase / tokPerGpuYear;
    // p99 wait estimate (decode only)
    const serviceTimeMs = config.latencyMs;
    const p99 = isPrefill ? null : p99WaitMs({
      rho: effectiveUtil, cServers: cServersInitial, serviceTimeMs,
    });
    return {
      gpuYears,
      sku: choice.sku,
      tp: config.tp, pp: config.pp, batch: config.batch,
      tpsPerGpu,
      latencyMs: config.latencyMs,
      bound: config.bound,
      utilization: effectiveUtil,
      utilCap,
      cServersEstimate: cServersInitial,
      p99WaitMs: p99,
      missedLatency: !!config.missedLatency,
      dollarsPerMtok: config.dollarsPerMtok,
    };
  }

  const prefill = gpuYearsForPhase(prefillChoice, prefillTokensPerYear, true);
  const decode  = gpuYearsForPhase(decodeChoice,  decodeTokensPerYear,  false);

  return { prefill, decode, specMul, kvBytesPerTok };
}

// Training & finetune: convert FLOP budget to GPU-years on whichever SKU is
// cheapest per useful FLOP after MFU.
function planFlopWorkload({ inputs, flopsPerYear, allowedSkus }) {
  if (flopsPerYear <= 0) return { gpuYears: 0, sku: null };
  let best = null;
  for (const sku of allowedSkus) {
    const gpu = GPUS[sku];
    if (!gpu) continue;
    const flopsPerGpuSec = (inputs.quantization === "fp8" && gpu.fp8_tflops > 0
      ? gpu.fp8_tflops : gpu.fp16_tflops) * 1e12 * (inputs.mfu || 0.35);
    const flopsPerGpuYear = flopsPerGpuSec * HOURS_PER_YEAR * 3600;
    const gpuYears = flopsPerYear / flopsPerGpuYear;
    const capexPerGpuYear = gpu.capex_usd / 3 + (gpu.power_w / 1000)
      * (inputs.pue || 1.25) * (inputs.electricityPrice || 0.07) * HOURS_PER_YEAR;
    const cost = gpuYears * capexPerGpuYear;
    if (!best || cost < best.cost) {
      best = { sku, gpu, gpuYears, cost, flopsPerGpuYear };
    }
  }
  return best || { gpuYears: Infinity, sku: null };
}

// ---------------------------------------------------------------------------
// Main entry: buildPlan
// ---------------------------------------------------------------------------

export function buildPlan(inputs) {
  const yearly = [];
  const horizonYears = inputs.horizonYears || 5;
  const allowedSkus = (inputs.allowedGpus && inputs.allowedGpus.length > 0)
    ? inputs.allowedGpus
    : Object.keys(GPUS);
  const electricityPrice = inputs.electricityPrice ?? 0.07;
  const pue = inputs.pue ?? 1.25;
  const inputLen = inputs.contextLen ?? 4000;
  const outputLen = inputs.outputLen ?? 200;
  const inputOutputRatio = inputs.inputOutputRatio;
  const effectiveInputLen = inputOutputRatio
    ? outputLen * inputOutputRatio
    : inputLen;

  for (let y = 0; y < horizonYears; y++) {
    const mGrow = modelGrowthFactor(inputs.modelGrowth ?? 0, y);
    const dGrow = demandGrowthFactor(inputs.demandGrowth ?? 0, y);
    const algMul = efficiencyMultiplier(inputs.algEfficiency ?? 0, y);
    const paramsBYear = (inputs.paramsB ?? 70) * mGrow;
    const kvBytesPerTok = resolveKvBytes(inputs, paramsBYear);

    // Per-workload planning
    const interactivePlan = planInferenceWorkload({
      inputs, year: y, paramsBYear, kvBytesPerTok,
      workloadKey: "interactive_inference",
      latencyMsTarget: LATENCY_TARGET_MS_PER_TOK.interactive_inference,
      ttftTargetMs: TTFT_TARGET_MS.interactive_inference,
      inputLen: effectiveInputLen, contextLen: inputs.contextLen ?? 4000, outputLen,
      tokensPerYear: (inputs.interactiveTokensPerDay ?? 0) * 365 * dGrow,
      electricityPrice, pue, algMul,
    });
    const batchPlan = planInferenceWorkload({
      inputs, year: y, paramsBYear, kvBytesPerTok,
      workloadKey: "batch_inference",
      latencyMsTarget: LATENCY_TARGET_MS_PER_TOK.batch_inference,
      ttftTargetMs: TTFT_TARGET_MS.batch_inference,
      inputLen: effectiveInputLen, contextLen: inputs.contextLen ?? 4000, outputLen,
      tokensPerYear: (inputs.batchTokensPerDay ?? 0) * 365 * dGrow,
      electricityPrice, pue, algMul,
    });
    const rlPlan = planInferenceWorkload({
      inputs, year: y, paramsBYear, kvBytesPerTok,
      workloadKey: "rl",
      latencyMsTarget: LATENCY_TARGET_MS_PER_TOK.rl,
      ttftTargetMs: TTFT_TARGET_MS.rl,
      inputLen: effectiveInputLen, contextLen: inputs.contextLen ?? 4000, outputLen,
      tokensPerYear: (inputs.rlTokensPerDay ?? 0) * 365 * dGrow,
      electricityPrice, pue, algMul,
    });

    // Training scales with model^2 × demand growth (Chinchilla-ish).
    const pretrainScale = Math.pow(mGrow, 2) * dGrow;
    const finetuneScale = mGrow * dGrow;
    const trainingPlan  = planFlopWorkload({
      inputs, allowedSkus,
      flopsPerYear: (inputs.pretrainFlopsPerYear ?? 0) * pretrainScale,
    });
    const finetunePlan  = planFlopWorkload({
      inputs, allowedSkus,
      flopsPerYear: (inputs.finetuneFlopsPerYear ?? 0) * finetuneScale,
    });

    // Aggregate per-workload GPU-years (legacy field for existing charts).
    const gpuYears = {
      interactive_inference: (interactivePlan.prefill.gpuYears || 0) + (interactivePlan.decode.gpuYears || 0),
      batch_inference:       (batchPlan.prefill.gpuYears || 0)       + (batchPlan.decode.gpuYears || 0),
      rl:                    (rlPlan.prefill.gpuYears || 0)          + (rlPlan.decode.gpuYears || 0),
      training:              trainingPlan.gpuYears || 0,
      finetune:              finetunePlan.gpuYears || 0,
    };
    const totalGpuYears = Object.values(gpuYears)
      .map((v) => isFinite(v) ? v : 0)
      .reduce((a, b) => a + b, 0);

    // GPUs by SKU: aggregate every (workload×phase) chosen SKU.
    const gpusBySku = {};
    function add(sku, gy) {
      if (!sku || !isFinite(gy)) return;
      gpusBySku[sku] = (gpusBySku[sku] || 0) + gy;
    }
    add(interactivePlan.prefill.sku, interactivePlan.prefill.gpuYears);
    add(interactivePlan.decode.sku,  interactivePlan.decode.gpuYears);
    add(batchPlan.prefill.sku,       batchPlan.prefill.gpuYears);
    add(batchPlan.decode.sku,        batchPlan.decode.gpuYears);
    add(rlPlan.prefill.sku,          rlPlan.prefill.gpuYears);
    add(rlPlan.decode.sku,           rlPlan.decode.gpuYears);
    add(trainingPlan.sku,            trainingPlan.gpuYears);
    add(finetunePlan.sku,            finetunePlan.gpuYears);

    // Convert GPU-years to peak GPU count via headroom factor.
    const headroom = inputs.headroomFactor ?? 1.25;
    const totalGpus = Math.ceil(totalGpuYears * headroom);

    // Owned vs rented split: rentalShare governs the ratio (legacy input).
    const rentalShare = inputs.rentalShare ?? 0.0;
    const ownedShare = 1 - rentalShare;
    const ownedGpus  = Math.ceil(totalGpus * ownedShare);
    const rentedGpus = totalGpus - ownedGpus;

    // Pick the most-common SKU as the "headline" GPU for legacy chart code.
    let headlineSku = null;
    let headlineCount = 0;
    for (const [sku, count] of Object.entries(gpusBySku)) {
      if (count > headlineCount) { headlineSku = sku; headlineCount = count; }
    }
    const headlineGpu = headlineSku ? GPUS[headlineSku] : GPUS[allowedSkus[0]];

    // Coherent fabric requirement for the year's frontier model on the headline SKU.
    const minScaleUpGpus = minGpusForModel({
      paramsB: paramsBYear,
      quantization: inputs.quantization,
      gpu: headlineGpu,
      kvHeadroomGb: 30,
    });

    // Power: weighted average across SKU mix.
    let totalKw = 0;
    for (const [sku, count] of Object.entries(gpusBySku)) {
      const gpu = GPUS[sku];
      if (!gpu) continue;
      totalKw += (count * headroom) * (gpu.power_w / 1000) * pue;
    }

    // Site placement
    const sitePlacement = placeMwAcrossRegions({
      mwNeededByYear: [totalKw / 1000],
      regions: inputs.regions || DEFAULT_REGIONS,
      preferredOrder: inputs.regionPreferredOrder,
      rentalShareOverride: rentalShare,
    });
    const siteThisYear = sitePlacement[0];

    // Latency-tier facility split (kept for backwards-compatible charts).
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

    // Annual energy
    const annualKwh = totalKw * HOURS_PER_YEAR;
    const electricityUsd = annualKwh * electricityPrice;

    // Refresh logic + CapEx attribution
    const refreshYears = inputs.refreshYears ?? 4;
    const isRefreshYear = (y === 0) || (y > 0 && y % refreshYears === 0);
    const incrementalGpus = y === 0 ? ownedGpus
      : Math.max(0, ownedGpus - (yearly[y - 1]?.ownedGpus ?? 0))
        + (isRefreshYear ? (yearly[y - refreshYears]?.ownedGpus ?? 0) : 0);

    // Average GPU CapEx weighted by SKU mix
    let weightedCapex = 0;
    let weightedNetwork = 0;
    let totalShare = 0;
    for (const [sku, count] of Object.entries(gpusBySku)) {
      const gpu = GPUS[sku];
      if (!gpu) continue;
      weightedCapex  += count * gpu.capex_usd;
      weightedNetwork += count * (NETWORK_COST_PER_GPU_USD[inputs.networkTier ?? "small_pod"] ?? 8000);
      totalShare += count;
    }
    const avgGpuCapex   = totalShare > 0 ? weightedCapex / totalShare : (headlineGpu?.capex_usd ?? 28000);
    const avgNetworkCap = totalShare > 0 ? weightedNetwork / totalShare : 8000;

    const gpuCapex      = incrementalGpus * avgGpuCapex;
    const networkCapex  = incrementalGpus * avgNetworkCap;
    const facilityCapex = isRefreshYear ? totalKw * DC.facility_capex_per_kw_usd : 0;
    const facilityOpex  = totalKw * DC.facility_capex_per_kw_usd * DC.facility_opex_pct_of_capex;

    // Rental cost: weighted by SKU mix in the rental pool too.
    let weightedRental = 0;
    let rentalShareSum = 0;
    for (const [sku, count] of Object.entries(gpusBySku)) {
      weightedRental += count * (RENTAL_PRICE_USD_PER_GPU_HOUR[sku] ?? 2.5);
      rentalShareSum += count;
    }
    const avgRentalRate = rentalShareSum > 0 ? weightedRental / rentalShareSum : 2.0;
    const rentalUsd = rentedGpus * avgRentalRate * HOURS_PER_YEAR;

    const totalCapex = gpuCapex + networkCapex + facilityCapex;
    const totalOpex  = electricityUsd + facilityOpex + rentalUsd;
    const yearTco    = totalCapex + totalOpex;
    const discounted = yearTco / Math.pow(1 + (inputs.discountRate ?? 0.10), y);

    // Internal $/Mtok: allocate TCO to inference workloads in proportion to
    // their GPU-years share, then divide by tokens served.
    const inferenceGpuYears = gpuYears.interactive_inference + gpuYears.batch_inference;
    const inferenceShare = totalGpuYears > 0 ? inferenceGpuYears / totalGpuYears : 0;
    const tokensServed = ((inputs.interactiveTokensPerDay ?? 0)
                       +  (inputs.batchTokensPerDay ?? 0)) * 365 * dGrow;
    const inferenceTco = yearTco * inferenceShare;
    const internalDollarsPerMtok = tokensServed > 0
      ? (inferenceTco * 1e6) / tokensServed : 0;
    const externalDollarsPerMtok = externalPricePerMtok(
      inputs.externalInitialPrice ?? 5.0,
      inputs.externalDeclineMult ?? 3.0,
      y,
    );

    yearly.push({
      year: y,
      yearParamsB: paramsBYear,
      kvBytesPerTok,
      // Per-workload, per-phase plans (NEW)
      allocations: {
        interactive_inference: interactivePlan,
        batch_inference: batchPlan,
        rl: rlPlan,
        training: trainingPlan,
        finetune: finetunePlan,
      },
      // Legacy aggregates (keep existing charts working)
      gpuYears,
      totalGpuYears,
      totalGpus,
      ownedGpus,
      rentedGpus,
      gpusBySku,                                     // NEW: SKU breakdown
      sitePlacement: siteThisYear,                   // NEW: regional placement
      siteShortfall: siteThisYear?.shortfall ?? 0,   // NEW
      ownedTierGpus,
      tierFacilityCount,
      numFacilities,
      minScaleUpGpus,
      headlineSku,
      totalKw,
      annualKwh,
      capex: { gpu: gpuCapex, network: networkCapex, facility: facilityCapex },
      opex:  { electricity: electricityUsd, facility: facilityOpex, rental: rentalUsd },
      yearTco,
      discounted,
      internalDollarsPerMtok,
      externalDollarsPerMtok,
      perfBound: {
        interactive: { prefill: interactivePlan.prefill, decode: interactivePlan.decode },
        batch:       { prefill: batchPlan.prefill,       decode: batchPlan.decode },
        rl:          { prefill: rlPlan.prefill,          decode: rlPlan.decode },
      },
    });
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
    totalDiscountedTco,
    peakGpus, peakMw, peakFacilities,
    gpu, headlineSku,
    anyShortfall,
  };
}

// ---------------------------------------------------------------------------
// Sensitivity & uncertainty (kept; signatures unchanged for charts.js)
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
    ["interactiveTokensPerDay", baseInputs.interactiveTokensPerDay * 0.5, baseInputs.interactiveTokensPerDay * 2.0],
    ["batchTokensPerDay",       baseInputs.batchTokensPerDay * 0.5,       baseInputs.batchTokensPerDay * 2.0],
    ["demandGrowth",            (baseInputs.demandGrowth ?? 0.5) - 0.15,  (baseInputs.demandGrowth ?? 0.5) + 0.15],
    ["modelGrowth",             (baseInputs.modelGrowth ?? 0.4) - 0.10,   (baseInputs.modelGrowth ?? 0.4) + 0.10],
    ["algEfficiency",           (baseInputs.algEfficiency ?? 0.5) - 0.20, (baseInputs.algEfficiency ?? 0.5) + 0.20],
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

export const _internal = { bytesPerParam, decodeThroughputPerGpu, prefillThroughputPerGpu };
