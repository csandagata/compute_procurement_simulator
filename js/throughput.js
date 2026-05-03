// Phase-aware throughput model and per-workload (TP, PP, batch, GPU) optimizer.
//
// Two phases:
//   - prefill: process the full input prompt in one pass. Compute-bound for
//     reasonable input lengths, latency = 2 * P_active * L_input / FLOPs.
//   - decode:  generate one output token at a time. Memory-bw-bound for small
//     batches, compute-bound at very large batch.
//
// Both share the same weights but typically run as separate clusters in
// disaggregated serving so each phase can pick its own (TP, batch, GPU).

import { approximateArchitecture } from "./kv.js";
import { specDecSpeedup } from "./spec_dec.js";

function bytesPerParam(quantization) {
  switch (quantization) {
    case "fp16": return 2;
    case "fp8":  return 1;
    case "int4": return 0.5;
    default:     return 2;
  }
}

function effectiveFlops(gpu, quantization) {
  if (quantization === "fp8" && gpu.fp8_tflops > 0) return gpu.fp8_tflops * 1e12;
  if (quantization === "int4" && gpu.fp8_tflops > 0) return gpu.fp8_tflops * 2 * 1e12;
  return gpu.fp16_tflops * 1e12;
}

// Network bandwidth per-GPU at the given degree of parallelism.
// Stays in scale-up domain (NVLink) for tp <= scaleup_domain_size, drops to
// scale-out (IB/RoCE) above that.
function netBandwidthBps(gpu, tp) {
  const scaleUp = gpu.scaleup_domain || 8;
  const inScaleUp = tp <= scaleUp;
  return (inScaleUp ? gpu.nvlink_gbs : gpu.scaleout_gbs) * 1e9;
}

// Decode-phase throughput per GPU, given (model, gpu, batch, tp).
export function decodeThroughputPerGpu({
  paramsB, activeFrac, quantization, gpu, batch, tp,
  kvBytesPerToken, contextLen, specDecMultiplier = 1,
}) {
  const params = paramsB * 1e9;
  const activeParams = params * activeFrac;
  const wpBytes = bytesPerParam(quantization);
  const flops = effectiveFlops(gpu, quantization);
  const hbmBw = gpu.hbm_bw_tbs * 1e12;
  const netBw = netBandwidthBps(gpu, tp);
  const arch = approximateArchitecture(paramsB);
  const { dModel, layers } = arch;

  // Time per output token (single-token forward pass on a batch of `batch` reqs).
  const tMem   = (wpBytes * params) / (tp * hbmBw);
  const tArith = (2 * activeParams * batch) / (tp * flops);
  const allreduceBytesPerLayer = 4 * dModel * 2 * batch;
  const tNet   = (allreduceBytesPerLayer * layers) / netBw
               + layers * 5e-6 * Math.log2(Math.max(2, tp));

  const tPerToken = Math.max(tMem, tArith, tNet);

  // HBM occupancy check: weights/tp + KV cache (sharded across tp).
  const weightBytesPerGpu = (wpBytes * params) / tp;
  const kvBytes = kvBytesPerToken * contextLen * batch;
  const hbmAvailBytes = gpu.hbm_gb * 1e9 * 0.85;
  const fitsInHbm = (weightBytesPerGpu + kvBytes / tp) < hbmAvailBytes;
  if (!fitsInHbm) return { tps: 0, tpsAggregate: 0, latencyMs: Infinity, bound: "OOM", fitsInHbm: false };

  // Per-GPU throughput is the aggregate batch divided across tp GPUs.
  const tpsAggregate = (batch / tPerToken) * specDecMultiplier;
  const tpsPerGpu = tpsAggregate / tp;
  const latencyMs = (tPerToken * 1000) / specDecMultiplier;
  let bound = "memory";
  if (tArith >= tMem && tArith >= tNet) bound = "compute";
  else if (tNet > tMem && tNet > tArith) bound = "network";
  return { tps: tpsPerGpu, tpsAggregate, latencyMs, bound, fitsInHbm };
}

// Prefill-phase throughput per GPU. Prefill processes `inputLen` tokens at once,
// so it's compute-bound: t_prefill ≈ 2 * activeParams * inputLen * batch / (tp*FLOPs).
// Memory cost is one weight read regardless of input length.
export function prefillThroughputPerGpu({
  paramsB, activeFrac, quantization, gpu, batch, tp, inputLen,
  kvBytesPerToken,
}) {
  const params = paramsB * 1e9;
  const activeParams = params * activeFrac;
  const wpBytes = bytesPerParam(quantization);
  const flops = effectiveFlops(gpu, quantization);
  const hbmBw = gpu.hbm_bw_tbs * 1e12;
  const netBw = netBandwidthBps(gpu, tp);
  const arch = approximateArchitecture(paramsB);
  const { dModel, layers } = arch;

  // Total tokens processed in this prefill: batch × inputLen.
  const tokens = batch * inputLen;
  const tCompute = (2 * activeParams * tokens) / (tp * flops);
  // Weights are loaded once, regardless of token count -> generally a small term.
  const tMem    = (wpBytes * params) / (tp * hbmBw);
  // Allreduce: scales with tokens through activations.
  const allreduceBytesPerLayer = 4 * dModel * 2 * tokens;
  const tNet    = (allreduceBytesPerLayer * layers) / netBw
                + layers * 5e-6 * Math.log2(Math.max(2, tp));
  // KV cache footprint after prefill = batch × inputLen × bytes/token, sharded across tp.
  const kvBytes = kvBytesPerToken * inputLen * batch;
  const weightBytesPerGpu = (wpBytes * params) / tp;
  const hbmAvailBytes = gpu.hbm_gb * 1e9 * 0.85;
  const fitsInHbm = (weightBytesPerGpu + kvBytes / tp) < hbmAvailBytes;
  if (!fitsInHbm) return { tps: 0, tpsAggregate: 0, latencyMs: Infinity, bound: "OOM", fitsInHbm: false };

  const totalTime = Math.max(tCompute, tMem, tNet);
  const tpsAggregate = tokens / totalTime;
  const tpsPerGpu = tpsAggregate / tp;
  const latencyMs = totalTime * 1000;
  let bound = "compute";
  if (tMem > tCompute && tMem >= tNet) bound = "memory";
  else if (tNet > tCompute && tNet > tMem) bound = "network";
  return { tps: tpsPerGpu, tpsAggregate, latencyMs, bound, fitsInHbm };
}

// Per-GPU $/Mtok used for ranking optimizer candidates. Power amortized over
// 3-year hardware life, plus electricity at PUE.
function gpuHourCost(gpu, electricityPrice, pue) {
  const capexPerHour = gpu.capex_usd / (3 * 365 * 24);
  const powerCostPerHour = (gpu.power_w / 1000) * pue * electricityPrice;
  return capexPerHour + powerCostPerHour;
}

// Search for the (TP, batch, PP) combination on a given GPU that minimises
// $/Mtok subject to a latency target and HBM fit. Returns best candidate or null.
//
// PP (pipeline parallel) is approximated by treating it as a multiplier on the
// effective scale-up domain: with PP=k, each pipeline stage holds 1/k of layers
// and gets 1/k weight bytes per GPU, but we pay an extra bubble overhead.
// We sweep PP only if the model doesn't fit at PP=1.
export function optimizeServingConfig({
  phase,                // "prefill" | "decode"
  paramsB, activeFrac, quantization, gpu,
  kvBytesPerToken, contextLen, inputLen,
  latencyMsTarget = Infinity,
  electricityPrice, pue,
  specDecMultiplier = 1,
  maxTp = null,
} = {}) {
  const scaleUp = gpu.scaleup_domain || 8;
  const tpCandidates = [1, 2, 4, 8, 16, 32, 64].filter((t) => t <= (maxTp || scaleUp * 2));
  const batchCandidates = phase === "prefill"
    ? [1, 2, 4, 8, 16, 32]
    : [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  const ppCandidates = [1, 2, 4];

  const cost = gpuHourCost(gpu, electricityPrice, pue);

  let best = null;
  for (const pp of ppCandidates) {
    for (const tp of tpCandidates) {
      const effectiveTp = tp * pp;        // total GPU domain holding the model
      // PP bubble penalty: ~1/PP utilization haircut for small batches; we just
      // multiply throughput by (PP / (PP + 1)) to capture pipeline bubbles.
      const ppUtil = pp === 1 ? 1 : pp / (pp + 1);
      for (const batch of batchCandidates) {
        let r;
        if (phase === "prefill") {
          r = prefillThroughputPerGpu({
            paramsB, activeFrac, quantization, gpu, batch, tp: effectiveTp,
            inputLen, kvBytesPerToken,
          });
        } else {
          r = decodeThroughputPerGpu({
            paramsB, activeFrac, quantization, gpu, batch, tp: effectiveTp,
            kvBytesPerToken, contextLen, specDecMultiplier,
          });
        }
        if (!r.fitsInHbm || r.tps <= 0) continue;
        const adjTps = r.tps * ppUtil;
        if (r.latencyMs > latencyMsTarget) continue;
        const dollarsPerMtok = (cost / 3600) / adjTps * 1e6;
        if (!best || dollarsPerMtok < best.dollarsPerMtok) {
          best = {
            tp, pp, effectiveTp, batch,
            tpsPerGpu: adjTps,
            latencyMs: r.latencyMs,
            bound: r.bound,
            dollarsPerMtok,
          };
        }
      }
    }
  }

  // If nothing met latency target, fall back to highest-throughput-that-fits.
  if (!best) {
    for (const pp of ppCandidates) {
      for (const tp of tpCandidates) {
        const effectiveTp = tp * pp;
        const ppUtil = pp === 1 ? 1 : pp / (pp + 1);
        for (const batch of batchCandidates) {
          let r;
          if (phase === "prefill") {
            r = prefillThroughputPerGpu({
              paramsB, activeFrac, quantization, gpu, batch, tp: effectiveTp,
              inputLen, kvBytesPerToken,
            });
          } else {
            r = decodeThroughputPerGpu({
              paramsB, activeFrac, quantization, gpu, batch, tp: effectiveTp,
              kvBytesPerToken, contextLen, specDecMultiplier,
            });
          }
          if (!r.fitsInHbm || r.tps <= 0) continue;
          const adjTps = r.tps * ppUtil;
          const dollarsPerMtok = (cost / 3600) / adjTps * 1e6;
          if (!best || adjTps > best.tpsPerGpu) {
            best = {
              tp, pp, effectiveTp, batch,
              tpsPerGpu: adjTps,
              latencyMs: r.latencyMs,
              bound: r.bound,
              dollarsPerMtok,
              missedLatency: true,
            };
          }
        }
      }
    }
  }
  return best;
}

// Given the GPU library and a workload spec, pick the cheapest GPU SKU that
// can serve this phase at its latency target. Returns the chosen SKU plus
// optimizer output. Restricted to SKUs in `allowedSkus` (array of keys).
export function pickBestSkuForPhase({
  gpus,                  // map: SKU key -> gpu spec
  allowedSkus,           // array of SKU keys
  phase,
  workloadSpec,          // contains paramsB, activeFrac, quantization, kvBytesPerToken, contextLen, inputLen, latencyMsTarget, specDecMultiplier
  electricityPrice, pue,
}) {
  let best = null;
  let bestSku = null;
  let bestConfig = null;
  for (const sku of allowedSkus) {
    const gpu = gpus[sku];
    if (!gpu) continue;
    const cfg = optimizeServingConfig({
      phase, gpu,
      ...workloadSpec,
      electricityPrice, pue,
    });
    if (!cfg) continue;
    if (!best || cfg.dollarsPerMtok < best.dollarsPerMtok) {
      best = cfg;
      bestSku = sku;
      bestConfig = cfg;
    }
  }
  if (!bestSku) return null;
  return { sku: bestSku, config: bestConfig };
}
