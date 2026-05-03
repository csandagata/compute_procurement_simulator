// KV-cache schemes. Determines bytes/token in the KV cache for the model.
// User can override with an explicit kvBytesPerToken; otherwise we derive from
// architecture. Numbers are dtype × (K + V) × heads/groups, summed across layers.

export const KV_SCHEMES = ["MHA", "GQA_8", "GQA_4", "MQA", "MLA"];

export const KV_SCHEME_LABELS = {
  MHA:   "Multi-head attention (full)",
  GQA_8: "Grouped-query attention (group=8)",
  GQA_4: "Grouped-query attention (group=4)",
  MQA:   "Multi-query attention",
  MLA:   "Multi-head latent attention (DeepSeek-style)",
};

// Architecture estimator: from total params -> approx (d_model, layers, heads, d_head).
// Same scaling we used in the throughput model so the two stay consistent.
export function approximateArchitecture(paramsB) {
  const dModel = 100 * Math.cbrt(paramsB);
  const layers = Math.max(8, Math.round(dModel / 96));
  const dHead = 128;
  const numQueryHeads = Math.max(8, Math.round(dModel / dHead));
  return { dModel, layers, dHead, numQueryHeads };
}

// Bytes per token in the KV cache, summed across all layers.
// dtypeBytes: usually 2 (fp16) for KV even with fp8 weights.
export function kvBytesPerToken({
  scheme = "GQA_8",
  paramsB,
  mlaCompressionDim = 576,    // DeepSeek-V3 default
  dtypeBytes = 2,
}) {
  const arch = approximateArchitecture(paramsB);
  const { layers, dHead, numQueryHeads } = arch;
  let perLayer;
  switch (scheme) {
    case "MHA":
      perLayer = 2 * numQueryHeads * dHead * dtypeBytes;
      break;
    case "GQA_8":
      perLayer = 2 * Math.max(1, Math.floor(numQueryHeads / 8)) * dHead * dtypeBytes;
      break;
    case "GQA_4":
      perLayer = 2 * Math.max(1, Math.floor(numQueryHeads / 4)) * dHead * dtypeBytes;
      break;
    case "MQA":
      perLayer = 2 * dHead * dtypeBytes;
      break;
    case "MLA":
      // MLA stores a compressed latent of size d_c per token, plus a small RoPE remainder.
      perLayer = (mlaCompressionDim + dHead / 2) * dtypeBytes;
      break;
    default:
      perLayer = 2 * numQueryHeads * dHead * dtypeBytes;
  }
  return perLayer * layers;
}
