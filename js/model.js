// Procurement simulation core.
//
// The math here is a deliberately simplified version of the inference-economics
// framework from Erdil/Epoch (https://arxiv.org/abs/2506.04645), restricted to
// what's useful for fleet planning rather than per-kernel scheduling.
//
// Headline relationships preserved:
//   - per-token decode latency is bounded below by the larger of memory-bw time
//     and compute time, plus a network/collective term that grows with TP degree
//   - throughput per GPU rises with batch until activation/network costs dominate
//   - scale-up fabric size needed to host a model = ceil(model_bytes / hbm_per_gpu)
//   - frontier model size grows over time, demand grows over time, alg efficiency
//     improves over time -> three superimposed exponentials drive fleet sizing
//
// Everything is a pure function. Inputs in, plan out.

import { GPUS, RENTAL_PRICE_USD_PER_GPU_HOUR, NETWORK_COST_PER_GPU_USD, DC } from "./gpus.js";
import {
  WORKLOAD_TYPES,
  LATENCY_TARGET_MS_PER_TOK,
  NEEDS_SCALE_UP_FABRIC,
  TIER_RTT_MS,
} from "./workloads.js";

const SECONDS_PER_YEAR = 3600 * 24 * 365;
const HOURS_PER_YEAR = 24 * 365;

// ---------------------------------------------------------------------------
// 1) Per-GPU throughput model (decode regime, batched)
// ---------------------------------------------------------------------------

// Activation-precision bytes per param while reading weights.
function bytesPerParam(quantization) {
  switch (quantization) {
    case "fp16": return 2;
    case "fp8":  return 1;
    case "int4": return 0.5;
    default:     return 2;
  }
}

// Effective FLOPs per GPU at this precision.
function effectiveFlops(gpu, quantization) {
  if (quantization === "fp8" && gpu.fp8_tflops > 0) return gpu.fp8_tflops * 1e12;
  if (quantization === "int4" && gpu.fp8_tflops > 0) return gpu.fp8_tflops * 2 * 1e12; // assume 2x via INT4
  return gpu.fp16_tflops * 1e12;
}

// Per-GPU decode tokens-per-second for a dense-equivalent model.
//   params:      total parameter count (dense-equivalent for MoE, weighted by sparsity)
//   activeFrac:  fraction of params touched per token (1.0 dense, 1/sparsity for MoE)
//   batch:       in-flight requests being decoded together
//   tp:          tensor-parallel degree across GPUs in scale-up domain
//   ctxKB:       average KV-cache footprint per request, in MB
//
// We compute three time terms per token and take the max:
//   t_mem  = total_weight_bytes / (TP * HBM_bw)        (weights streamed once per token)
//   t_arith = 2 * activeParams * batch / (TP * FLOPs)  (FLOPs to do the matmul)
//   t_net  = (4 * d_model * batch * 2 * layers) / NetBW + collective_latency
// We approximate d_model from params^(1/3) scaling and layers from params^(1/3).
export function decodeTokensPerSecondPerGpu({
  paramsB,
  activeFrac,
  quantization,
  gpu,
  batch,
  tp,
  kvBytesPerToken,
  contextLen,
}) {
  const params = paramsB * 1e9;
  const activeParams = params * activeFrac;
  const wpBytes = bytesPerParam(quantization);
  const flops = effectiveFlops(gpu, quantization);
  const hbmBw = gpu.hbm_bw_tbs * 1e12;
  const netBw = (tp <= (gpu.scaleup_domain || 8)) ? gpu.nvlink_gbs * 1e9 : gpu.scaleout_gbs * 1e9;

  // Approximate transformer shape from total params (dense-equivalent).
  // d_model ~ k * P^(1/3); layers ~ d_model/96. These are rough but capture
  // network-cost scaling faithfully.
  const dModel = 100 * Math.cbrt(params / 1e9);
  const layers = Math.max(8, Math.round(dModel / 96));

  const tMem    = (wpBytes * params) / (tp * hbmBw);
  const tArith  = (2 * activeParams * batch) / (tp * flops);
  const allreduceBytesPerLayerPerToken = 4 * dModel * 2 * batch; // 4 reductions, fp16 activations
  const tNet    = (allreduceBytesPerLayerPerToken * layers) / netBw + layers * 5e-6 * Math.log2(Math.max(2, tp));

  // KV cache claim against HBM
  const kvBytes = kvBytesPerToken * contextLen * batch;
  const weightBytesPerGpu = (wpBytes * params) / tp;
  const hbmAvail = gpu.hbm_gb * 1e9;
  const fitsInHbm = (weightBytesPerGpu + kvBytes / tp) < hbmAvail * 0.85;

  const tPerToken = Math.max(tMem, tArith, tNet);
  if (!fitsInHbm) return { tps: 0, latencyMs: Infinity, bound: "OOM" };

  // tps is per-GPU output rate; we share batch tokens across the tp GPUs working together
  const tpsAggregate = batch / tPerToken;
  const tpsPerGpu = tpsAggregate / tp;
  const latencyMs = tPerToken * 1000;

  let bound = "memory";
  if (tArith > tMem && tArith >= tNet) bound = "compute";
  else if (tNet > tMem && tNet > tArith) bound = "network";

  return { tps: tpsPerGpu, tpsAggregate, latencyMs, bound, fitsInHbm };
}

// Sweep batch sizes to find the (batch, tps) point that meets a latency target
// at the highest throughput. Mirrors the Pareto-front search in the Erdil paper
// but coarse: 16 batch grid points is enough for planning.
export function bestBatchForLatency(spec, latencyMsTarget) {
  const batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  let best = { batch: 1, tps: 0, latencyMs: Infinity, bound: "n/a" };
  for (const b of batches) {
    const r = decodeTokensPerSecondPerGpu({ ...spec, batch: b });
    if (r.latencyMs <= latencyMsTarget && r.tps > best.tps) {
      best = { batch: b, ...r };
    }
  }
  // If no batch met latency, return the largest-throughput one anyway (with a flag).
  if (best.tps === 0) {
    for (const b of batches) {
      const r = decodeTokensPerSecondPerGpu({ ...spec, batch: b });
      if (r.tps > best.tps && r.fitsInHbm !== false) {
        best = { batch: b, ...r, missedLatency: true };
      }
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// 2) Year-over-year scaling functions
// ---------------------------------------------------------------------------

export function modelGrowthFactor(rate, year) { return Math.pow(1 + rate, year); }
export function demandGrowthFactor(rate, year) { return Math.pow(1 + rate, year); }
// Algorithmic-efficiency multiplier improves throughput per GPU-hour.
export function efficiencyMultiplier(rate, year) { return Math.pow(1 + rate, year); }

// External $/Mtok declines with the Epoch-observed rate (50x/yr median historically,
// but flattening as we approach physical limits; we use a user-tunable rate).
export function externalPricePerMtok(initialUsd, declineMultPerYear, year) {
  return initialUsd / Math.pow(declineMultPerYear, year);
}

// ---------------------------------------------------------------------------
// 3) Scale-up fabric sizing (smallest pod that can host the model)
// ---------------------------------------------------------------------------

export function minGpusForModel({ paramsB, quantization, gpu, kvHeadroomGb }) {
  const wpBytes = bytesPerParam(quantization);
  const weightGb = (paramsB * 1e9 * wpBytes) / 1e9;
  const usableHbm = gpu.hbm_gb * 0.85 - kvHeadroomGb;
  if (usableHbm <= 0) return Infinity;
  return Math.max(1, Math.ceil(weightGb / usableHbm));
}

// ---------------------------------------------------------------------------
// 4) Workload -> GPU-hours/year demand
// ---------------------------------------------------------------------------

// Auto-scale TP up to whatever power-of-2 is needed for weights to fit.
function tpForFit({ paramsB, quantization, gpu, baseTp }) {
  const minFit = minGpusForModel({ paramsB, quantization, gpu, kvHeadroomGb: 30 });
  // Round up to next power of two for clean tensor-parallel splits.
  let tp = Math.max(baseTp, minFit);
  let p2 = 1;
  while (p2 < tp) p2 *= 2;
  return p2;
}

// Returns GPU-hours/year required to serve this inference workload
// at the latency target with the chosen GPU/quantization.
function gpuHoursForInferenceWorkload({
  tokensPerYear, gpu, paramsB, activeFrac, quantization,
  contextLen, kvBytesPerToken, latencyMsPerTok, tp, utilization,
}) {
  if (tokensPerYear <= 0) {
    return { gpuHours: 0, perf: { tps: 0, latencyMs: 0, bound: "n/a", tp } };
  }
  const effTp = tpForFit({ paramsB, quantization, gpu, baseTp: tp });
  const r = bestBatchForLatency(
    { paramsB, activeFrac, quantization, gpu, tp: effTp, kvBytesPerToken, contextLen },
    latencyMsPerTok,
  );
  if (!r || r.tps <= 0) {
    // Model still won't run even with auto-scaled TP — flag as infeasible.
    return { gpuHours: Infinity, perf: { ...(r || {}), tp: effTp, infeasible: true } };
  }
  // Effective tokens/GPU-hour after utilization haircut (kernel gaps, traffic shaping, idle).
  const effectiveTokPerGpuHour = r.tps * 3600 * utilization;
  const gpuHours = tokensPerYear / effectiveTokPerGpuHour;
  return { gpuHours, perf: { ...r, tp: effTp } };
}

// Training & finetune: convert a FLOP budget into GPU-years.
function gpuHoursForFlopBudget({ flopsPerYear, gpu, quantization, mfu }) {
  const flopsPerGpuSec = effectiveFlops(gpu, quantization) * mfu;
  return flopsPerYear / flopsPerGpuSec / 3600;
}

// ---------------------------------------------------------------------------
// 5) Top-level annual plan builder
// ---------------------------------------------------------------------------

export function buildPlan(inputs) {
  const {
    horizonYears,
    gpuKey,
    quantization,
    paramsB,
    activeFrac,
    contextLen,
    kvBytesPerToken,
    interactiveTokensPerDay,
    batchTokensPerDay,
    rlTokensPerDay,
    pretrainFlopsPerYear,
    finetuneFlopsPerYear,
    tpDegree,
    mfu,
    utilization,
    demandGrowth,
    modelGrowth,
    algEfficiency,
    refreshYears,
    discountRate,
    externalInitialPrice,
    externalDeclineMult,
    edgeFacilityShare,
    regionalFacilityShare,
    centralFacilityShare,
    facilityMaxMw,
    headroomFactor,
    rentalShare,
    pue,
    electricityPrice,
    networkTier,
  } = inputs;

  const gpu = GPUS[gpuKey];
  const yearly = [];

  for (let y = 0; y < horizonYears; y++) {
    const mGrow = modelGrowthFactor(modelGrowth, y);
    const dGrow = demandGrowthFactor(demandGrowth, y);
    const algMul = efficiencyMultiplier(algEfficiency, y);
    const yearParamsB = paramsB * mGrow;

    // Inference-style workloads: scale tokens/day by demand growth, divide effective
    // throughput per GPU by alg efficiency multiplier (more tokens per GPU-hour over time).
    const inferenceCommon = {
      gpu, paramsB: yearParamsB, activeFrac, quantization,
      contextLen, kvBytesPerToken, tp: tpDegree, utilization: utilization * algMul,
    };

    const interactive = gpuHoursForInferenceWorkload({
      ...inferenceCommon,
      tokensPerYear: interactiveTokensPerDay * 365 * dGrow,
      latencyMsPerTok: LATENCY_TARGET_MS_PER_TOK.interactive_inference,
    });
    const batchInfer = gpuHoursForInferenceWorkload({
      ...inferenceCommon,
      tokensPerYear: batchTokensPerDay * 365 * dGrow,
      latencyMsPerTok: LATENCY_TARGET_MS_PER_TOK.batch_inference,
    });
    const rl = gpuHoursForInferenceWorkload({
      ...inferenceCommon,
      tokensPerYear: rlTokensPerDay * 365 * dGrow,
      latencyMsPerTok: LATENCY_TARGET_MS_PER_TOK.rl,
    });

    // Training / finetune: FLOP-budget driven. Pretraining FLOPs scale with model^2
    // approximately (Chinchilla-ish: tokens ~ P, FLOPs ~ 6*P*tokens ~ P^2).
    const pretrainScale = Math.pow(mGrow, 2) * dGrow;
    const finetuneScale = mGrow * dGrow;

    const trainHours = gpuHoursForFlopBudget({
      flopsPerYear: pretrainFlopsPerYear * pretrainScale,
      gpu, quantization, mfu,
    });
    const finetuneHours = gpuHoursForFlopBudget({
      flopsPerYear: finetuneFlopsPerYear * finetuneScale,
      gpu, quantization, mfu,
    });

    const gpuYears = {
      interactive_inference: interactive.gpuHours / HOURS_PER_YEAR,
      batch_inference: batchInfer.gpuHours / HOURS_PER_YEAR,
      rl: rl.gpuHours / HOURS_PER_YEAR,
      training: trainHours / HOURS_PER_YEAR,
      finetune: finetuneHours / HOURS_PER_YEAR,
    };

    const totalGpuYears = Object.values(gpuYears).reduce((a, b) => a + b, 0);
    const totalGpus = Math.ceil(totalGpuYears * headroomFactor);

    // Scale-up fabric size to host the year's frontier model.
    const minScaleUpGpus = minGpusForModel({
      paramsB: yearParamsB, quantization, gpu, kvHeadroomGb: 20,
    });

    // Facility plan
    const ownedShare = 1 - rentalShare;
    const ownedGpus = Math.ceil(totalGpus * ownedShare);
    const rentedGpus = totalGpus - ownedGpus;
    const ownedTierGpus = {
      edge: Math.ceil(ownedGpus * edgeFacilityShare),
      regional: Math.ceil(ownedGpus * regionalFacilityShare),
      central: Math.ceil(ownedGpus * centralFacilityShare),
    };
    const tierFacilityCount = {};
    for (const [tier, count] of Object.entries(ownedTierGpus)) {
      const mwPerGpu = (gpu.power_w / 1000) * pue / 1000;
      const mwTotal = count * mwPerGpu;
      tierFacilityCount[tier] = Math.max(count > 0 ? 1 : 0, Math.ceil(mwTotal / facilityMaxMw));
    }
    const numFacilities = Object.values(tierFacilityCount).reduce((a, b) => a + b, 0);

    // Power & energy
    const totalKw = ownedGpus * (gpu.power_w / 1000) * pue;
    const annualKwh = totalKw * HOURS_PER_YEAR;
    const electricityUsd = annualKwh * electricityPrice;

    // CapEx (only for newly-acquired GPUs that year, naive: all in year 0,
    // refresh wave every refreshYears).
    const isRefreshYear = (y === 0) || (y > 0 && y % refreshYears === 0);
    const incrementalGpus =
      y === 0 ? ownedGpus
      : isRefreshYear ? Math.max(0, ownedGpus - (yearly[y - refreshYears]?.ownedGpus ?? 0)) + (yearly[y - refreshYears]?.ownedGpus ?? 0)
      : Math.max(0, ownedGpus - (yearly[y - 1]?.ownedGpus ?? 0));

    const gpuCapex = incrementalGpus * gpu.capex_usd;
    const networkCapex = incrementalGpus * NETWORK_COST_PER_GPU_USD[networkTier];
    const facilityCapex = isRefreshYear ? totalKw * DC.facility_capex_per_kw_usd : 0;
    const facilityOpex = totalKw * DC.facility_capex_per_kw_usd * DC.facility_opex_pct_of_capex;

    // Rental cost
    const rentalUsd = rentedGpus * RENTAL_PRICE_USD_PER_GPU_HOUR[gpuKey] * HOURS_PER_YEAR;

    const totalCapex = gpuCapex + networkCapex + facilityCapex;
    const totalOpex = electricityUsd + facilityOpex + rentalUsd;
    const yearTco = totalCapex + totalOpex;
    const discounted = yearTco / Math.pow(1 + discountRate, y);

    // Internal $/Mtok: allocate TCO to inference workloads in proportion to their
    // GPU-year share. This is the right "marginal cost of serving a token" comparison
    // against external API pricing (which doesn't include the customer's pretrain CapEx).
    const inferenceGpuYears = gpuYears.interactive_inference + gpuYears.batch_inference;
    const inferenceShare = totalGpuYears > 0 ? inferenceGpuYears / totalGpuYears : 0;
    const tokensServed = (interactiveTokensPerDay + batchTokensPerDay) * 365 * dGrow;
    const inferenceTco = yearTco * inferenceShare;
    const internalDollarsPerMtok = tokensServed > 0 ? (inferenceTco * 1e6) / tokensServed : 0;
    const externalDollarsPerMtok = externalPricePerMtok(externalInitialPrice, externalDeclineMult, y);

    yearly.push({
      year: y,
      gpuYears,
      totalGpuYears,
      totalGpus,
      ownedGpus,
      rentedGpus,
      ownedTierGpus,
      tierFacilityCount,
      numFacilities,
      minScaleUpGpus,
      totalKw,
      annualKwh,
      perfBound: {
        interactive: interactive.perf,
        batch: batchInfer.perf,
        rl: rl.perf,
      },
      capex: { gpu: gpuCapex, network: networkCapex, facility: facilityCapex },
      opex: { electricity: electricityUsd, facility: facilityOpex, rental: rentalUsd },
      yearTco,
      discounted,
      internalDollarsPerMtok,
      externalDollarsPerMtok,
      yearParamsB,
    });
  }

  // Aggregate KPIs
  const totalDiscountedTco = yearly.reduce((a, b) => a + b.discounted, 0);
  const peakGpus = Math.max(...yearly.map((y) => y.totalGpus));
  const peakMw = Math.max(...yearly.map((y) => y.totalKw / 1000));
  const peakFacilities = Math.max(...yearly.map((y) => y.numFacilities));

  return { yearly, totalDiscountedTco, peakGpus, peakMw, peakFacilities, gpu };
}

// ---------------------------------------------------------------------------
// 6) Sensitivity analysis (tornado)
// ---------------------------------------------------------------------------

export function tornado(baseInputs, lever, lo, hi) {
  const loInputs = { ...baseInputs, [lever]: lo };
  const hiInputs = { ...baseInputs, [lever]: hi };
  const loPlan = buildPlan(loInputs);
  const hiPlan = buildPlan(hiInputs);
  return {
    lever,
    lo: loPlan.totalDiscountedTco,
    hi: hiPlan.totalDiscountedTco,
  };
}

// Run a battery of ±X% perturbations and return tornado data.
export function tornadoBattery(baseInputs, baseTco) {
  const levers = [
    ["interactiveTokensPerDay", 0.5, 2.0],
    ["batchTokensPerDay", 0.5, 2.0],
    ["demandGrowth", baseInputs.demandGrowth - 0.15, baseInputs.demandGrowth + 0.15],
    ["modelGrowth", baseInputs.modelGrowth - 0.10, baseInputs.modelGrowth + 0.10],
    ["algEfficiency", baseInputs.algEfficiency - 0.20, baseInputs.algEfficiency + 0.20],
    ["mfu", Math.max(0.05, baseInputs.mfu - 0.15), Math.min(0.7, baseInputs.mfu + 0.15)],
    ["utilization", Math.max(0.2, baseInputs.utilization - 0.2), Math.min(0.95, baseInputs.utilization + 0.2)],
    ["pretrainFlopsPerYear", baseInputs.pretrainFlopsPerYear * 0.5, baseInputs.pretrainFlopsPerYear * 2],
    ["electricityPrice", baseInputs.electricityPrice * 0.5, baseInputs.electricityPrice * 1.5],
    ["rentalShare", Math.max(0, baseInputs.rentalShare - 0.25), Math.min(1, baseInputs.rentalShare + 0.25)],
  ];
  return levers.map(([k, lo, hi]) => {
    const r = tornado(baseInputs, k, lo, hi);
    return { lever: k, lo: r.lo - baseTco, hi: r.hi - baseTco, loVal: lo, hiVal: hi };
  });
}

// ---------------------------------------------------------------------------
// 7) Demand-uncertainty fan (Monte-Carlo-lite)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 8) Latency-cost Pareto frontier (single-year snapshot)
// ---------------------------------------------------------------------------

export function latencyCostFrontier(inputs, year = 0) {
  const gpu = GPUS[inputs.gpuKey];
  const yearParamsB = inputs.paramsB * Math.pow(1 + inputs.modelGrowth, year);
  const points = [];
  // Sweep latency targets and TP degrees; record the best $/Mtok at each.
  const latencyTargets = [10, 20, 30, 50, 75, 100, 150, 250, 500, 1000];
  const tps = [1, 2, 4, 8, 16, 32, 64];
  for (const lat of latencyTargets) {
    let bestPrice = Infinity;
    let bestTps = 0;
    let bestPoint = null;
    for (const tp of tps) {
      const r = bestBatchForLatency({
        paramsB: yearParamsB,
        activeFrac: inputs.activeFrac,
        quantization: inputs.quantization,
        gpu, tp,
        kvBytesPerToken: inputs.kvBytesPerToken,
        contextLen: inputs.contextLen,
      }, lat);
      if (r.tps <= 0) continue;
      const tokPerGpuHour = r.tps * 3600 * inputs.utilization;
      const gpuHourCost = gpu.capex_usd / (HOURS_PER_YEAR * 3) + gpu.power_w / 1000 * inputs.pue * inputs.electricityPrice;
      const usdPerMtok = (gpuHourCost * 1e6) / tokPerGpuHour;
      if (usdPerMtok < bestPrice) {
        bestPrice = usdPerMtok;
        bestTps = r.tps;
        bestPoint = { tp, batch: r.batch, latencyMs: r.latencyMs, bound: r.bound };
      }
    }
    if (isFinite(bestPrice)) {
      points.push({ targetLatencyMs: lat, dollarsPerMtok: bestPrice, tpsPerGpu: bestTps, ...bestPoint });
    }
  }
  return points;
}

export const _internal = { effectiveFlops, bytesPerParam };
