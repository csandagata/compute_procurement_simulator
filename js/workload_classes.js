// Workload-class library.
//
// Each workload class declares enough about itself for the planner to:
//   - dispatch the right throughput formula (memory-bw vs compute vs FP64)
//   - filter GPU SKUs that can run it (precision, ecosystem, FP64 requirement)
//   - apply the right growth model (LLM scaling-law / campaign / steady / latent)
//   - aggregate into the legacy fleet-by-workload chart category
//
// Several AI-for-science classes have *unfitted* scaling laws — we don't
// actually know yet whether a domain like multi-omics or fusion-plasma ML
// will turn out to be data-bound (where more compute won't help much) or
// compute-bound (where the next 10x of compute unlocks meaningful gains).
// The planner exposes this as an explicit `scalingCase` per workload:
//
//   - "low":    pessimistic — domain is data-bound or saturates quickly
//   - "median": central-estimate growth as published / observed today
//   - "high":   optimistic — domain turns out to be compute-bound
//   - "fitted": (LLM-only) use derived Hoffmann-curve scaling
//
// This lets a buyer ask: "if AlphaFold-class compute turns out to be
// compute-bound and scales 5x/yr, what fleet would I need?" alongside
// "if it's data-bound and stays flat, what does that look like?"

export const DOMAINS = {
  llm:           { label: "Large language models",       color: "#4F8EF7" },
  biology:       { label: "Biology / chemistry",          color: "#7BB661" },
  physics:       { label: "Physics / energy",             color: "#E08E45" },
  earth_obs:     { label: "Earth observation / climate",  color: "#3FA7D6" },
  industry:      { label: "Industry / digital twins",     color: "#9F86C0" },
  defense_intel: { label: "Defense / intelligence",       color: "#C8553D" },
};

export const COMPUTE_PATTERNS = [
  "autoregressive",          // LLM-style decode + prefill
  "single_pass_inference",   // one forward pass per unit (AlphaFold, classifier)
  "iterative_simulation",    // PDE/N-body time-stepping (CFD, MD)
  "active_learning_loop",    // bursty inference + retraining (materials, prot. design)
  "continuous_training",     // pretraining or large training run
];

export const SCALING_CASES = ["low", "median", "high", "fitted"];

// Library of workload archetypes. Numbers are illustrative starting points;
// every entry's flops_per_unit, throughput_yoy, etc. is adjustable in the UI.
export const WORKLOAD_CLASSES = {

  // ----------------------------- LLM family -----------------------------

  llm_interactive: {
    id: "llm_interactive",
    label: "Interactive LLM serving (chat, copilot)",
    domain: "llm",
    compute_pattern: "autoregressive",
    bound_default: "memory",          // decode is memory-BW-bound
    precision_options: ["fp8", "fp16", "int4"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "edge",
    parallelism: "tp_pp",
    autoregressive: true,
    benefits_from_spec_dec: true,
    legacy_category: "interactive_inference",
    unit: "token",
    default_demand: { unitsPerYear: 5e9 * 365, label: "Output tokens/year" },
    growth_model_default: "scaling_law",
    scaling_uncertainty: {
      low:    { yoy: 0.30 },         // conservative: alg efficiency mostly cancels demand growth
      median: { yoy: 0.60 },
      high:   { yoy: 1.20 },         // viral product growth + agentic workflows
      fitted: { yoy: null },         // use Hoffmann-curve scaling-law engine
    },
    notes: "Memory-bandwidth-bound decode; prefill is compute-bound. Scaling laws fitted (Hoffmann/Kaplan).",
  },

  llm_batch: {
    id: "llm_batch",
    label: "Batch LLM (summarisation, eval, ETL)",
    domain: "llm",
    compute_pattern: "autoregressive",
    bound_default: "memory",
    precision_options: ["fp8", "int4", "fp16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "central",
    parallelism: "tp_pp",
    autoregressive: true,
    benefits_from_spec_dec: true,
    legacy_category: "batch_inference",
    unit: "token",
    default_demand: { unitsPerYear: 20e9 * 365, label: "Output tokens/year" },
    growth_model_default: "scaling_law",
    scaling_uncertainty: {
      low: { yoy: 0.20 }, median: { yoy: 0.50 }, high: { yoy: 1.00 }, fitted: { yoy: null },
    },
    notes: "Throughput-optimised; same physics as interactive, different latency contract.",
  },

  llm_pretrain: {
    id: "llm_pretrain",
    label: "LLM pretraining campaigns",
    domain: "llm",
    compute_pattern: "continuous_training",
    bound_default: "compute",
    precision_options: ["fp8", "fp16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "central",
    parallelism: "3d",
    autoregressive: false,
    benefits_from_spec_dec: false,
    legacy_category: "training",
    unit: "flop",
    default_demand: { unitsPerYear: 1e23, label: "Pretrain FLOPs/year" },
    growth_model_default: "scaling_law",
    scaling_uncertainty: {
      low: { yoy: 0.50 }, median: { yoy: 1.50 }, high: { yoy: 3.00 }, fitted: { yoy: null },
    },
    notes: "Chinchilla-aligned: FLOPs ≈ 6·P·D, D ≈ 20·P → C ~ P². Drives the 'compute doubles every 6 months' line.",
  },

  llm_finetune: {
    id: "llm_finetune",
    label: "LLM fine-tuning / continual training",
    domain: "llm",
    compute_pattern: "continuous_training",
    bound_default: "compute",
    precision_options: ["fp8", "fp16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "regional",
    parallelism: "tp_pp",
    legacy_category: "finetune",
    unit: "flop",
    default_demand: { unitsPerYear: 5e22, label: "Finetune FLOPs/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.10 }, median: { yoy: 0.40 }, high: { yoy: 1.00 }, fitted: { yoy: null },
    },
  },

  llm_rl_rollout: {
    id: "llm_rl_rollout",
    label: "RL post-training rollouts (verifier-coupled)",
    domain: "llm",
    compute_pattern: "autoregressive",
    bound_default: "memory",
    precision_options: ["fp8", "fp16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "regional",
    parallelism: "tp_pp",
    autoregressive: true,
    benefits_from_spec_dec: true,
    legacy_category: "rl",
    unit: "token",
    default_demand: { unitsPerYear: 0, label: "Rollout tokens/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.30 }, median: { yoy: 1.00 }, high: { yoy: 3.00 }, fitted: { yoy: null },
    },
    notes: "Bursty: rollout cluster + reward model cluster + training cluster, often with feedback loops.",
  },

  // --------------------------- Biology / chemistry ---------------------------

  alphafold_screening: {
    id: "alphafold_screening",
    label: "Protein structure prediction (AlphaFold-class)",
    domain: "biology",
    compute_pattern: "single_pass_inference",
    bound_default: "compute",
    precision_options: ["fp16", "bf16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "central",
    parallelism: "embarrassing",
    legacy_category: "batch_inference",
    unit: "structure",
    flops_per_unit: 3e15,                 // ~3 PFLOPs per ~600-residue structure (AF2 ballpark)
    training_inference_ratio: 0.02,        // train rarely; mostly inference
    default_demand: { unitsPerYear: 1e8, label: "Structures predicted/year" },
    growth_model_default: "latent",
    scaling_uncertainty: {
      low:    { yoy: 0.10 },              // saturates: comparable to current accuracy ceiling
      median: { yoy: 0.50 },              // expand to design + multimers
      high:   { yoy: 2.00 },              // compute-bound regime → physics-quality predictions
    },
    notes: "Compute-bound forward pass; embarrassingly parallel; scaling law NOT yet fitted (data may bound).",
  },

  protein_design_rl: {
    id: "protein_design_rl",
    label: "De novo protein/molecule design (RFdiffusion + RL)",
    domain: "biology",
    compute_pattern: "active_learning_loop",
    bound_default: "compute",
    precision_options: ["fp16", "bf16"],
    fp64_required: false,
    ecosystem: "cuda",
    latency_tier: "regional",
    parallelism: "embarrassing",
    legacy_category: "rl",
    unit: "design",
    flops_per_unit: 1e15,
    training_inference_ratio: 0.10,
    default_demand: { unitsPerYear: 1e7, label: "Designs/year" },
    growth_model_default: "campaign",
    growth_params: { campaignsPerYear: 6, gpuWeeksPerCampaign: 200, growthYoy: 0.5 },
    scaling_uncertainty: {
      low: { yoy: 0.20 }, median: { yoy: 0.80 }, high: { yoy: 3.00 },
    },
    notes: "Active-learning loops: design → wet-lab feedback → retrain. Bursty, campaign-shaped.",
  },

  drug_docking: {
    id: "drug_docking",
    label: "High-throughput virtual screening / docking",
    domain: "biology",
    compute_pattern: "single_pass_inference",
    bound_default: "compute",
    precision_options: ["fp16", "bf16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "central",
    parallelism: "embarrassing",
    legacy_category: "batch_inference",
    unit: "pose",
    flops_per_unit: 5e13,                 // ~50 TFLOPs per pose (DiffDock-class)
    training_inference_ratio: 0.01,
    default_demand: { unitsPerYear: 1e10, label: "Poses/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.20 }, median: { yoy: 0.80 }, high: { yoy: 2.50 },
    },
    notes: "Throughput-driven; cheap per unit; benefits from cheap commodity GPUs.",
  },

  genomics_inference: {
    id: "genomics_inference",
    label: "Genomics inference (variant calling, DeepVariant)",
    domain: "biology",
    compute_pattern: "single_pass_inference",
    bound_default: "compute",
    precision_options: ["fp16", "fp32"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "central",
    parallelism: "embarrassing",
    legacy_category: "batch_inference",
    unit: "variant",
    flops_per_unit: 5e10,
    training_inference_ratio: 0.005,
    default_demand: { unitsPerYear: 5e10, label: "Variants/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.20 }, median: { yoy: 0.50 }, high: { yoy: 1.50 },
    },
    notes: "Sequencing volume drives demand; throughput more than latency.",
  },

  multi_omics_integration: {
    id: "multi_omics_integration",
    label: "Multi-omics integration models",
    domain: "biology",
    compute_pattern: "single_pass_inference",
    bound_default: "memory",
    precision_options: ["fp16", "fp32"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "regional",
    parallelism: "tp_pp",
    legacy_category: "batch_inference",
    unit: "patient",
    flops_per_unit: 1e14,
    training_inference_ratio: 0.20,        // retrain often as new modalities added
    default_demand: { unitsPerYear: 1e7, label: "Patients/year" },
    growth_model_default: "latent",
    scaling_uncertainty: {
      low:    { yoy: 0.05 },              // strongly data-bound: limited multi-modal datasets
      median: { yoy: 0.30 },
      high:   { yoy: 1.50 },              // if foundation-model-style scaling emerges
    },
    notes: "Heavily data-bound today; could become compute-bound if multi-modal foundation models work.",
  },

  ml_force_field_md: {
    id: "ml_force_field_md",
    label: "ML force fields for molecular dynamics (NequIP, MACE)",
    domain: "biology",
    compute_pattern: "iterative_simulation",
    bound_default: "compute",
    precision_options: ["fp32", "fp16"],
    fp64_required: false,                  // ML force fields can run FP32
    ecosystem: "cuda_or_rocm",
    latency_tier: "central",
    parallelism: "weak_scale",
    legacy_category: "batch_inference",
    unit: "ns_simulated",
    flops_per_unit: 1e15,                  // 1 PFLOP per ns simulated (rough)
    training_inference_ratio: 0.10,
    default_demand: { unitsPerYear: 1e6, label: "ns of MD simulated/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.20 }, median: { yoy: 0.60 }, high: { yoy: 2.00 },
    },
  },

  // ----------------------------- Physics / energy -----------------------------

  cfd_digital_twin: {
    id: "cfd_digital_twin",
    label: "CFD / fluid dynamics digital twin",
    domain: "physics",
    compute_pattern: "iterative_simulation",
    bound_default: "fp64",
    precision_options: ["fp64"],
    fp64_required: true,
    ecosystem: "hpc",
    latency_tier: "central",
    parallelism: "strong_scale",
    legacy_category: "training",         // bucket as training-equivalent for legacy charts
    unit: "timestep",
    flops_per_unit: 1e18,                // industrial-resolution timestep (ballpark)
    training_inference_ratio: 0,
    default_demand: { unitsPerYear: 1e6, label: "FP64 timesteps/year" },
    growth_model_default: "linear",
    growth_params: { yoyAddPercent: 0.15 },
    scaling_uncertainty: {
      low: { yoy: 0.05 }, median: { yoy: 0.15 }, high: { yoy: 0.50 },
    },
    notes: "Strongly FP64-bound; B200 is a poor fit; MI300X / H100 / CPUs more appropriate.",
  },

  fusion_plasma_sim: {
    id: "fusion_plasma_sim",
    label: "Fusion plasma simulation (gyrokinetic + ML hybrid)",
    domain: "physics",
    compute_pattern: "iterative_simulation",
    bound_default: "fp64",
    precision_options: ["fp64", "fp32"],
    fp64_required: true,
    ecosystem: "hpc",
    latency_tier: "central",
    parallelism: "strong_scale",
    legacy_category: "training",
    unit: "shot_simulated",
    flops_per_unit: 5e19,                // PFLOP-hours per simulation shot
    training_inference_ratio: 0.05,
    default_demand: { unitsPerYear: 5e3, label: "Shots simulated/year" },
    growth_model_default: "latent",
    scaling_uncertainty: {
      low:    { yoy: 0.10 },
      median: { yoy: 0.50 },
      high:   { yoy: 3.00 },             // if ML surrogates unlock 100x more parameter scans
    },
    notes: "FP64 + ML surrogate hybrid; data-rich (decades of plasma diagnostics) but physics-bounded.",
  },

  particle_event_reco: {
    id: "particle_event_reco",
    label: "Particle physics event reconstruction (LHC-class)",
    domain: "physics",
    compute_pattern: "single_pass_inference",
    bound_default: "compute",
    precision_options: ["fp16", "fp32"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "central",
    parallelism: "embarrassing",
    legacy_category: "batch_inference",
    unit: "event",
    flops_per_unit: 5e10,                // ~50 GFLOPs per event (CNN/GNN-based reco)
    training_inference_ratio: 0.02,
    default_demand: { unitsPerYear: 5e10, label: "Events reconstructed/year" },
    growth_model_default: "linear",
    growth_params: { yoyAddPercent: 0.10 },
    scaling_uncertainty: {
      low: { yoy: 0.05 }, median: { yoy: 0.15 }, high: { yoy: 0.50 },
    },
    notes: "HL-LHC drives 10x event rate by 2030; otherwise modest growth.",
  },

  astrophysics_sim: {
    id: "astrophysics_sim",
    label: "Astrophysical N-body / hydro simulations",
    domain: "physics",
    compute_pattern: "iterative_simulation",
    bound_default: "fp64",
    precision_options: ["fp64"],
    fp64_required: true,
    ecosystem: "hpc",
    latency_tier: "central",
    parallelism: "strong_scale",
    legacy_category: "training",
    unit: "snapshot",
    flops_per_unit: 1e19,
    training_inference_ratio: 0,
    default_demand: { unitsPerYear: 1e3, label: "Simulation snapshots/year" },
    growth_model_default: "linear",
    growth_params: { yoyAddPercent: 0.10 },
    scaling_uncertainty: {
      low: { yoy: 0.02 }, median: { yoy: 0.10 }, high: { yoy: 0.40 },
    },
  },

  // -------------------- Earth observation / climate --------------------

  weather_graphnet: {
    id: "weather_graphnet",
    label: "ML weather/climate models (GraphCast, Pangu)",
    domain: "earth_obs",
    compute_pattern: "iterative_simulation",
    bound_default: "compute",
    precision_options: ["fp16", "bf16", "fp32"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "regional",
    parallelism: "weak_scale",
    legacy_category: "batch_inference",
    unit: "forecast",
    flops_per_unit: 1e16,                // global forecast inference
    training_inference_ratio: 0.20,
    default_demand: { unitsPerYear: 1e5, label: "Global forecasts/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.20 }, median: { yoy: 0.60 }, high: { yoy: 2.00 },
    },
    notes: "Replacing classical NWP; resolution + ensemble size drive growth.",
  },

  climate_classical: {
    id: "climate_classical",
    label: "Classical climate models (CMIP, IFS)",
    domain: "earth_obs",
    compute_pattern: "iterative_simulation",
    bound_default: "fp64",
    precision_options: ["fp64", "fp32"],
    fp64_required: true,
    ecosystem: "hpc",
    latency_tier: "central",
    parallelism: "strong_scale",
    legacy_category: "training",
    unit: "century_simulated",
    flops_per_unit: 1e21,
    training_inference_ratio: 0,
    default_demand: { unitsPerYear: 100, label: "Simulated centuries/year" },
    growth_model_default: "linear",
    growth_params: { yoyAddPercent: 0.10 },
    scaling_uncertainty: {
      low: { yoy: 0.05 }, median: { yoy: 0.10 }, high: { yoy: 0.30 },
    },
  },

  satellite_imagery: {
    id: "satellite_imagery",
    label: "Satellite imagery (SAR/optical, segmentation, change detection)",
    domain: "earth_obs",
    compute_pattern: "single_pass_inference",
    bound_default: "compute",
    precision_options: ["fp16", "int8", "bf16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "regional",
    parallelism: "embarrassing",
    legacy_category: "batch_inference",
    unit: "scene",
    flops_per_unit: 1e12,                // ~1 TFLOP per high-res scene
    training_inference_ratio: 0.05,
    default_demand: { unitsPerYear: 1e8, label: "Scenes processed/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.30 }, median: { yoy: 0.80 }, high: { yoy: 2.50 },
    },
    notes: "Constellation growth + per-scene model size both compound. INT8 friendly.",
  },

  // ------------------------- Industry / digital twins -------------------------

  manufacturing_digital_twin: {
    id: "manufacturing_digital_twin",
    label: "Manufacturing digital twin (FEA + ML hybrid)",
    domain: "industry",
    compute_pattern: "iterative_simulation",
    bound_default: "fp64",
    precision_options: ["fp64", "fp32"],
    fp64_required: true,
    ecosystem: "hpc",
    latency_tier: "regional",
    parallelism: "strong_scale",
    legacy_category: "training",
    unit: "scenario",
    flops_per_unit: 1e17,
    training_inference_ratio: 0.10,
    default_demand: { unitsPerYear: 1e4, label: "Scenarios/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.10 }, median: { yoy: 0.40 }, high: { yoy: 1.50 },
    },
  },

  materials_active_learning: {
    id: "materials_active_learning",
    label: "Materials discovery (DFT + ML active learning)",
    domain: "industry",
    compute_pattern: "active_learning_loop",
    bound_default: "fp64",
    precision_options: ["fp64", "fp32"],
    fp64_required: true,
    ecosystem: "hpc",
    latency_tier: "central",
    parallelism: "embarrassing",
    legacy_category: "batch_inference",
    unit: "candidate",
    flops_per_unit: 5e15,                // average across DFT + ML force-field steps
    training_inference_ratio: 0.15,
    default_demand: { unitsPerYear: 1e6, label: "Candidates/year" },
    growth_model_default: "campaign",
    growth_params: { campaignsPerYear: 4, gpuWeeksPerCampaign: 500, growthYoy: 0.4 },
    scaling_uncertainty: {
      low: { yoy: 0.10 }, median: { yoy: 0.50 }, high: { yoy: 2.00 },
    },
  },

  network_digital_twin: {
    id: "network_digital_twin",
    label: "Network/grid/supply-chain digital twin",
    domain: "industry",
    compute_pattern: "iterative_simulation",
    bound_default: "compute",
    precision_options: ["fp32", "fp16"],
    fp64_required: false,
    ecosystem: "cuda_or_rocm",
    latency_tier: "regional",
    parallelism: "weak_scale",
    legacy_category: "batch_inference",
    unit: "scenario",
    flops_per_unit: 1e14,
    training_inference_ratio: 0.10,
    default_demand: { unitsPerYear: 1e5, label: "Scenarios/year" },
    growth_model_default: "exponential",
    scaling_uncertainty: {
      low: { yoy: 0.10 }, median: { yoy: 0.40 }, high: { yoy: 1.20 },
    },
  },
};

// Group classes by domain for UI rendering.
export function workloadsByDomain() {
  const out = {};
  for (const [id, w] of Object.entries(WORKLOAD_CLASSES)) {
    if (!out[w.domain]) out[w.domain] = [];
    out[w.domain].push(w);
  }
  return out;
}

// Helper: map a workload class -> legacy category for backwards-compatible charts.
export function legacyCategoryOf(workloadId) {
  return WORKLOAD_CLASSES[workloadId]?.legacy_category || "batch_inference";
}
