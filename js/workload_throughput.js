// Workload-class-aware throughput dispatcher.
//
// Different compute patterns need different throughput formulas:
//   - autoregressive:       LLM-style, falls through to throughput.js
//   - single_pass_inference: forward pass per unit, compute-bound
//   - iterative_simulation:  PDE/N-body, often FP64-bound, weak/strong scaling
//   - active_learning_loop:  bursty mix of inference + retraining
//   - continuous_training:   FLOP-budget driven, MFU-haircut
//
// Each returns the same shape: { sku, gpuYears, perGpuRate, bound, perfDetails }.

import { GPUS } from "./gpus.js";
import { optimizeServingConfig } from "./throughput.js";

const HOURS_PER_YEAR = 24 * 365;

function flopsForWorkload(gpu, workload, mfu) {
  const bound = workload.bound_default;
  if (bound === "fp64" && workload.fp64_required) {
    return (gpu.fp64_tflops || 0) * 1e12 * mfu;
  }
  // For compute-bound non-FP64 workloads, pick the highest precision the
  // workload can use that the GPU supports.
  const accepts = workload.precision_options || ["fp16"];
  if (accepts.includes("fp8") && (gpu.fp8_tflops || 0) > 0) return gpu.fp8_tflops * 1e12 * mfu;
  if (accepts.includes("int4") && (gpu.fp8_tflops || 0) > 0) return gpu.fp8_tflops * 2 * 1e12 * mfu;
  if (accepts.includes("fp16") || accepts.includes("bf16")) return gpu.fp16_tflops * 1e12 * mfu;
  if (accepts.includes("fp32")) return (gpu.fp32_tflops || gpu.fp16_tflops / 4) * 1e12 * mfu;
  if (accepts.includes("fp64")) return (gpu.fp64_tflops || 0) * 1e12 * mfu;
  return gpu.fp16_tflops * 1e12 * mfu;
}

// Filter SKUs that can run this workload (vendor / FP64 / ecosystem constraints).
export function eligibleSkus(workload, allowedSkus) {
  const out = [];
  for (const sku of allowedSkus) {
    const gpu = GPUS[sku];
    if (!gpu) continue;
    if (workload.fp64_required && (!gpu.is_hpc_capable || (gpu.fp64_tflops || 0) < 30)) continue;
    if (workload.ecosystem === "cuda" && gpu.vendor !== "nvidia") continue;
    if (workload.ecosystem === "hpc" && !gpu.is_hpc_capable) continue;
    out.push(sku);
  }
  return out.length > 0 ? out : allowedSkus.slice(); // fall back to all if no eligible
}

function gpuHourCost(gpu, electricityPrice, pue) {
  const capexPerHour = gpu.capex_usd / (3 * HOURS_PER_YEAR);
  const powerCostPerHour = (gpu.power_w / 1000) * pue * electricityPrice;
  return capexPerHour + powerCostPerHour;
}

// --- Autoregressive (LLM-style): delegate to existing prefill+decode optimiser ---
function planAutoregressive({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul }) {
  // Tokens/year for the LLM workloads.
  const inputLen = ctx.contextLen ?? 4000;
  const outputLen = ctx.outputLen ?? 200;
  const prefillShare = inputLen / (inputLen + outputLen);
  const decodeShare = 1 - prefillShare;

  function chooseSku(phase, latencyTarget) {
    let best = null;
    for (const sku of allowedSkus) {
      const gpu = GPUS[sku];
      if (!gpu) continue;
      const cfg = optimizeServingConfig({
        phase,
        paramsB: ctx.paramsBYear, activeFrac: ctx.activeFrac, quantization: ctx.quantization,
        gpu,
        kvBytesPerToken: ctx.kvBytesPerTok,
        contextLen: ctx.contextLen, inputLen,
        latencyMsTarget: latencyTarget,
        electricityPrice: ctx.electricityPrice, pue: ctx.pue,
        specDecMultiplier: phase === "decode" ? ctx.specMul : 1.0,
        maxTp: ctx.maxTp,
      });
      if (!cfg) continue;
      if (!best || cfg.dollarsPerMtok < best.config.dollarsPerMtok) {
        best = { sku, gpu, config: cfg };
      }
    }
    return best;
  }

  const decode = chooseSku("decode", ctx.decodeLatencyMs);
  const prefill = chooseSku("prefill", ctx.prefillLatencyMs);

  function gyForPhase(choice, tokens) {
    if (!choice) return { gpuYears: Infinity, infeasible: true };
    const tps = choice.config.tpsPerGpu * algMul;
    if (tps <= 0) return { gpuYears: Infinity, infeasible: true };
    const tokPerYr = tps * 3600 * 24 * 365 * (ctx.utilization || 0.6);
    return {
      gpuYears: tokens / tokPerYr,
      sku: choice.sku,
      tp: choice.config.tp, pp: choice.config.pp, batch: choice.config.batch,
      tpsPerGpu: tps, latencyMs: choice.config.latencyMs,
      bound: choice.config.bound, dollarsPerMtok: choice.config.dollarsPerMtok,
    };
  }

  const prefillResult = gyForPhase(prefill, demandUnitsPerYear * prefillShare);
  const decodeResult  = gyForPhase(decode,  demandUnitsPerYear * decodeShare);

  return {
    pattern: "autoregressive",
    prefill: prefillResult,
    decode: decodeResult,
    gpuYears: (prefillResult.gpuYears || 0) + (decodeResult.gpuYears || 0),
    sku: decodeResult.sku || prefillResult.sku,
  };
}

// --- Single-pass inference (AlphaFold, satellite imagery, particle reco) ---
function planSinglePassInference({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul }) {
  const skus = eligibleSkus(workload, allowedSkus);
  let best = null;
  for (const sku of skus) {
    const gpu = GPUS[sku];
    if (!gpu) continue;
    const flopsPerSec = flopsForWorkload(gpu, workload, ctx.mfu || 0.35);
    if (flopsPerSec <= 0) continue;
    const unitsPerSec = flopsPerSec / (workload.flops_per_unit || 1e15);
    const unitsPerYr = unitsPerSec * 3600 * 24 * 365 * (ctx.utilization || 0.6) * algMul;
    if (unitsPerYr <= 0) continue;
    const gpuYears = demandUnitsPerYear / unitsPerYr;
    const cost = gpuYears * (gpu.capex_usd + gpu.power_w / 1000 * ctx.pue * ctx.electricityPrice * HOURS_PER_YEAR);
    if (!best || cost < best.cost) {
      best = {
        pattern: "single_pass_inference",
        sku, gpu, gpuYears, cost,
        unitsPerSec, unitsPerYr,
        bound: workload.bound_default,
        flopsPerSec, flopsPerUnit: workload.flops_per_unit,
      };
    }
  }
  return best || { pattern: "single_pass_inference", gpuYears: Infinity, infeasible: true };
}

// --- Iterative simulation (CFD, fusion plasma, climate, MD) ---
function planIterativeSimulation({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul }) {
  // Same dispatch as single-pass for now, but with strong-scale parallel haircut
  // (Amdahl-like): doubling GPUs gives roughly 1.7x throughput at this granularity.
  const skus = eligibleSkus(workload, allowedSkus);
  const parallelEfficiency = workload.parallelism === "strong_scale" ? 0.7
                           : workload.parallelism === "weak_scale" ? 0.9
                           : 1.0;
  let best = null;
  for (const sku of skus) {
    const gpu = GPUS[sku];
    if (!gpu) continue;
    const flopsPerSec = flopsForWorkload(gpu, workload, ctx.mfu || 0.35) * parallelEfficiency;
    if (flopsPerSec <= 0) continue;
    const unitsPerSec = flopsPerSec / (workload.flops_per_unit || 1e18);
    const unitsPerYr = unitsPerSec * 3600 * 24 * 365 * (ctx.utilization || 0.6) * algMul;
    if (unitsPerYr <= 0) continue;
    const gpuYears = demandUnitsPerYear / unitsPerYr;
    const cost = gpuYears * (gpu.capex_usd + gpu.power_w / 1000 * ctx.pue * ctx.electricityPrice * HOURS_PER_YEAR);
    if (!best || cost < best.cost) {
      best = {
        pattern: "iterative_simulation",
        sku, gpu, gpuYears, cost,
        unitsPerSec, unitsPerYr,
        bound: workload.bound_default,
        parallelEfficiency,
      };
    }
  }
  return best || { pattern: "iterative_simulation", gpuYears: Infinity, infeasible: true };
}

// --- Active-learning loop (campaigns of inference + retraining) ---
function planActiveLearningLoop({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul }) {
  // Treat as single-pass inference but with extra training overhead.
  const inferenceResult = planSinglePassInference({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul });
  if (inferenceResult.infeasible) return inferenceResult;
  // Training compute as fraction of inference compute -> additive GPU-years.
  const trainingShare = workload.training_inference_ratio || 0.05;
  const trainingGpuYears = inferenceResult.gpuYears * trainingShare;
  return {
    ...inferenceResult,
    pattern: "active_learning_loop",
    inferenceGpuYears: inferenceResult.gpuYears,
    trainingGpuYears,
    gpuYears: inferenceResult.gpuYears + trainingGpuYears,
  };
}

// --- Continuous training (FLOP-budget driven) ---
function planContinuousTraining({ workload, allowedSkus, demandUnitsPerYear, ctx }) {
  if (demandUnitsPerYear <= 0) return { pattern: "continuous_training", gpuYears: 0, sku: null };
  const skus = eligibleSkus(workload, allowedSkus);
  let best = null;
  for (const sku of skus) {
    const gpu = GPUS[sku];
    if (!gpu) continue;
    const flopsPerSec = flopsForWorkload(gpu, workload, ctx.mfu || 0.35);
    if (flopsPerSec <= 0) continue;
    const flopsPerYear = flopsPerSec * HOURS_PER_YEAR * 3600;
    const gpuYears = demandUnitsPerYear / flopsPerYear;
    const cost = gpuYears * (gpu.capex_usd + gpu.power_w / 1000 * ctx.pue * ctx.electricityPrice * HOURS_PER_YEAR);
    if (!best || cost < best.cost) {
      best = { pattern: "continuous_training", sku, gpu, gpuYears, cost,
               flopsPerSec, flopsPerYear,
               bound: workload.bound_default };
    }
  }
  return best || { pattern: "continuous_training", gpuYears: Infinity, infeasible: true };
}

// Top-level dispatcher.
export function planWorkload({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul = 1 }) {
  switch (workload.compute_pattern) {
    case "autoregressive":
      return planAutoregressive({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul });
    case "single_pass_inference":
      return planSinglePassInference({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul });
    case "iterative_simulation":
      return planIterativeSimulation({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul });
    case "active_learning_loop":
      return planActiveLearningLoop({ workload, allowedSkus, demandUnitsPerYear, ctx, algMul });
    case "continuous_training":
      return planContinuousTraining({ workload, allowedSkus, demandUnitsPerYear, ctx });
    default:
      return { gpuYears: 0, sku: null, pattern: "unknown" };
  }
}
