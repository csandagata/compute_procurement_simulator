// Workload archetypes. Each maps a high-level demand intent to the parameters
// the procurement model needs (token volumes, latency targets, training compute, etc.).

export const WORKLOAD_TYPES = ["interactive_inference", "batch_inference", "training", "rl", "finetune"];

export const WORKLOAD_LABELS = {
  interactive_inference: "Interactive inference (chat, copilot)",
  batch_inference: "Batch inference (summarization, eval, ETL)",
  training: "Pretraining / large training runs",
  rl: "RL post-training (rollouts + verifier)",
  finetune: "Fine-tuning / continual training",
};

// Latency targets per workload (per-output-token, ms).
// Interactive needs <50ms/tok to feel snappy; batch can tolerate >>500ms/tok if throughput-optimized.
export const LATENCY_TARGET_MS_PER_TOK = {
  interactive_inference: 30,
  batch_inference: 500,
  training: Infinity,
  rl: 100,                 // rollouts benefit from speed but not as strict
  finetune: Infinity,
};

// Whether the workload requires a coherent (scale-up) fabric or scale-out is fine.
export const NEEDS_SCALE_UP_FABRIC = {
  interactive_inference: true,    // tensor parallel for low latency on big models
  batch_inference: false,         // can pipeline across cheaper interconnect
  training: true,                 // 3D parallelism wants high BW domain
  rl: true,
  finetune: true,
};

// Minimum facility latency tier policy (ms RTT to user).
// Used to decide how many edge facilities are needed.
export const TIER_RTT_MS = {
  edge: 20,
  regional: 60,
  central: 200,
};
