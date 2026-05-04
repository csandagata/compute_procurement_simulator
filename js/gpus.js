// GPU / accelerator reference database.
//
// Three vintage layers:
//   - Shipping today (A100, H100/H200, B200, GB200, MI300X, TPU v5p/v6e, Gaudi3, ...)
//   - Announced / near-term (Rubin R100 2026, MI355X 2025, TPU v7 Ironwood 2025)
//   - Speculative (Rubin Ultra 2027, Feynman 2028) — flagged as such; perf/W
//     extrapolated at ~2x per year per Nvidia's published cadence
//
// Numbers are public spec sheets where available; speculative entries use
// vendor roadmaps and historical scaling. Treat post-2026 figures as
// planning placeholders.
//
// New fields vs. v1:
//   - vendor: nvidia | amd | google | intel | cerebras | groq | sambanova
//   - ecosystem_maturity: 0..1 throughput haircut. 1.0 = full CUDA/cuDNN/TRT-LLM
//     stack at peak MFU; lower for less-mature stacks where production code paths
//     don't yet hit theoretical FLOPs. The optimizer multiplies usable FLOPs by
//     this, preventing "cheapest spec sheet wins" from being a dominant strategy.
//   - availability: merchant | gcp_only | doe_only | limited | speculative
//   - is_hpc_capable, fp64_tflops: as before
//   - status: shipping | announced | speculative

// Throughput haircut applied to nominal FLOPs. Calibrates against widely-
// reported real-world utilization differences between stacks; tune freely.
function eco(v) { return v; }

export const GPUS = {
  // ============================== A100 family ==============================
  A100_80GB: {
    label: "NVIDIA A100 80GB SXM",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(1.00),
    availability: "merchant", status: "shipping", intro_year: 2020,
    fp16_tflops: 312, fp8_tflops: 0, fp32_tflops: 19.5, fp64_tflops: 19.5,
    hbm_gb: 80, hbm_bw_tbs: 2.0,
    nvlink_gbs: 600, scaleout_gbs: 200, scaleup_domain: 8,
    power_w: 400, capex_usd: 12000,
    is_hpc_capable: true,
    notes: "Widely available secondary market. Powers Polaris, Perlmutter, many academic clusters.",
  },

  // ============================== Hopper family ==============================
  H100_SXM: {
    label: "NVIDIA H100 SXM 80GB",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(1.00),
    availability: "merchant", status: "shipping", intro_year: 2023,
    fp16_tflops: 989, fp8_tflops: 1979, fp32_tflops: 67, fp64_tflops: 67,
    hbm_gb: 80, hbm_bw_tbs: 3.35,
    nvlink_gbs: 900, scaleout_gbs: 400, scaleup_domain: 8,
    power_w: 700, capex_usd: 28000, is_hpc_capable: true,
    notes: "Mature production stack (TRT-LLM, FlashAttention, vLLM). Default choice for LLM inference.",
  },
  H200_SXM: {
    label: "NVIDIA H200 SXM 141GB",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(1.00),
    availability: "merchant", status: "shipping", intro_year: 2024,
    fp16_tflops: 989, fp8_tflops: 1979, fp32_tflops: 67, fp64_tflops: 67,
    hbm_gb: 141, hbm_bw_tbs: 4.8,
    nvlink_gbs: 900, scaleout_gbs: 400, scaleup_domain: 8,
    power_w: 700, capex_usd: 32000, is_hpc_capable: true,
    notes: "H100 with bigger/faster HBM. Helps long-context and decode bandwidth.",
  },

  // ============================== Blackwell family ==============================
  B200: {
    label: "NVIDIA B200 SXM",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(0.95),
    availability: "merchant", status: "shipping", intro_year: 2025,
    fp16_tflops: 2250, fp8_tflops: 4500, fp32_tflops: 80, fp64_tflops: 37,
    hbm_gb: 192, hbm_bw_tbs: 8.0,
    nvlink_gbs: 1800, scaleout_gbs: 800, scaleup_domain: 8,
    power_w: 1000, capex_usd: 40000, is_hpc_capable: false,
    notes: "AI-optimised: FP4/FP8 strong, FP64 intentionally downgraded. Poor fit for CFD/fusion.",
  },
  GB200_NVL72: {
    label: "NVIDIA GB200 (NVL72 rack-scale)",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(0.90),
    availability: "merchant", status: "shipping", intro_year: 2025,
    fp16_tflops: 2500, fp8_tflops: 5000, fp32_tflops: 90, fp64_tflops: 45,
    hbm_gb: 192, hbm_bw_tbs: 8.0,
    nvlink_gbs: 1800, scaleout_gbs: 800, scaleup_domain: 72,
    power_w: 1200, capex_usd: 50000, is_hpc_capable: false,
    notes: "Coherent 72-GPU NVLink domain — best for very large LLMs needing scale-up parallelism.",
  },

  // ============================== Rubin family (announced/projected) ==============================
  R100: {
    label: "NVIDIA R100 (Rubin) [announced]",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(0.85),
    availability: "merchant", status: "announced", intro_year: 2026,
    fp16_tflops: 4500, fp8_tflops: 9000, fp32_tflops: 160, fp64_tflops: 70,
    hbm_gb: 288, hbm_bw_tbs: 13.0,    // HBM4 8-stack
    nvlink_gbs: 3600, scaleout_gbs: 1600, scaleup_domain: 144,
    power_w: 1400, capex_usd: 55000, is_hpc_capable: true,
    notes: "Announced for 2026. NVL144 rack scale-up. ~2× B200 perf at +40% power. Numbers are vendor-projected.",
  },
  R200_RubinUltra: {
    label: "NVIDIA R200 (Rubin Ultra) [projected]",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(0.75),
    availability: "merchant", status: "speculative", intro_year: 2027,
    fp16_tflops: 9000, fp8_tflops: 18000, fp32_tflops: 320, fp64_tflops: 100,
    hbm_gb: 384, hbm_bw_tbs: 21.0,
    nvlink_gbs: 5000, scaleout_gbs: 3200, scaleup_domain: 576,
    power_w: 1800, capex_usd: 75000, is_hpc_capable: true,
    notes: "Speculative 2027 follow-on. NVL576 rack-scale. Use as a planning placeholder for buy-now-vs-wait analysis.",
  },
  F100_Feynman: {
    label: "NVIDIA F100 (Feynman) [speculative]",
    vendor: "nvidia", ecosystem: "cuda", ecosystem_maturity: eco(0.70),
    availability: "merchant", status: "speculative", intro_year: 2028,
    fp16_tflops: 18000, fp8_tflops: 36000, fp32_tflops: 600, fp64_tflops: 180,
    hbm_gb: 512, hbm_bw_tbs: 32.0,
    nvlink_gbs: 8000, scaleout_gbs: 6400, scaleup_domain: 1024,
    power_w: 2200, capex_usd: 100000, is_hpc_capable: true,
    notes: "Pure roadmap extrapolation (2× per gen on a 12-month cycle). Useful for 5-year planning placeholders.",
  },

  // ============================== AMD Instinct ==============================
  MI300X: {
    label: "AMD MI300X 192GB",
    vendor: "amd", ecosystem: "rocm", ecosystem_maturity: eco(0.78),
    availability: "merchant", status: "shipping", intro_year: 2024,
    fp16_tflops: 1300, fp8_tflops: 2600, fp32_tflops: 163, fp64_tflops: 81,
    hbm_gb: 192, hbm_bw_tbs: 5.3,
    nvlink_gbs: 896, scaleout_gbs: 400, scaleup_domain: 8,
    power_w: 750, capex_usd: 18000, is_hpc_capable: true,
    notes: "Best-in-class FP64 + cheap. ROCm software stack still maturing — 22% MFU haircut applied.",
  },
  MI325X: {
    label: "AMD MI325X 256GB",
    vendor: "amd", ecosystem: "rocm", ecosystem_maturity: eco(0.78),
    availability: "merchant", status: "shipping", intro_year: 2024,
    fp16_tflops: 1300, fp8_tflops: 2600, fp32_tflops: 163, fp64_tflops: 81,
    hbm_gb: 256, hbm_bw_tbs: 6.0,
    nvlink_gbs: 896, scaleout_gbs: 400, scaleup_domain: 8,
    power_w: 1000, capex_usd: 22000, is_hpc_capable: true,
    notes: "MI300X refresh with bigger/faster HBM3e.",
  },
  MI355X: {
    label: "AMD MI355X (CDNA4) [announced]",
    vendor: "amd", ecosystem: "rocm", ecosystem_maturity: eco(0.75),
    availability: "merchant", status: "announced", intro_year: 2025,
    fp16_tflops: 2300, fp8_tflops: 4600, fp32_tflops: 350, fp64_tflops: 130,
    hbm_gb: 288, hbm_bw_tbs: 8.0,
    nvlink_gbs: 1800, scaleout_gbs: 800, scaleup_domain: 8,
    power_w: 1000, capex_usd: 30000, is_hpc_capable: true,
    notes: "CDNA4. AMD's Blackwell competitor; aggressive FP4/FP6 push. ROCm haircut still applies.",
  },
  MI300A: {
    label: "AMD MI300A APU (El Capitan)",
    vendor: "amd", ecosystem: "rocm", ecosystem_maturity: eco(0.80),
    availability: "doe_only", status: "shipping", intro_year: 2024,
    fp16_tflops: 980, fp8_tflops: 1960, fp32_tflops: 122, fp64_tflops: 61,
    hbm_gb: 128, hbm_bw_tbs: 5.3,
    nvlink_gbs: 896, scaleout_gbs: 400, scaleup_domain: 4,
    power_w: 760, capex_usd: 0,            // not generally available
    is_hpc_capable: true,
    notes: "Integrated CPU+GPU APU powering El Capitan (LLNL). Listed for completeness; not generally procurable.",
  },

  // ============================== Google TPU ==============================
  TPU_v5p: {
    label: "Google TPU v5p (Cloud only)",
    vendor: "google", ecosystem: "jax_xla", ecosystem_maturity: eco(0.92),
    availability: "gcp_only", status: "shipping", intro_year: 2023,
    fp16_tflops: 459, fp8_tflops: 459, fp32_tflops: 459, fp64_tflops: 0,
    hbm_gb: 95, hbm_bw_tbs: 2.76,
    nvlink_gbs: 4800, scaleout_gbs: 800, scaleup_domain: 256,    // ICI 8x16x16 tori
    power_w: 500, capex_usd: 0, is_hpc_capable: false,
    notes: "Rentable on GCP only; no FP64. Excellent JAX/XLA stack maturity.",
  },
  TPU_v6e_Trillium: {
    label: "Google TPU v6e (Trillium) [GCP only]",
    vendor: "google", ecosystem: "jax_xla", ecosystem_maturity: eco(0.92),
    availability: "gcp_only", status: "shipping", intro_year: 2024,
    fp16_tflops: 918, fp8_tflops: 1836, fp32_tflops: 918, fp64_tflops: 0,
    hbm_gb: 32, hbm_bw_tbs: 1.64,
    nvlink_gbs: 3584, scaleout_gbs: 800, scaleup_domain: 256,
    power_w: 400, capex_usd: 0, is_hpc_capable: false,
    notes: "Inference-tuned variant. Cheaper per token but less HBM than v5p.",
  },
  TPU_v7_Ironwood: {
    label: "Google TPU v7 (Ironwood) [GCP only, 2025]",
    vendor: "google", ecosystem: "jax_xla", ecosystem_maturity: eco(0.85),
    availability: "gcp_only", status: "shipping", intro_year: 2025,
    fp16_tflops: 2300, fp8_tflops: 4614, fp32_tflops: 2300, fp64_tflops: 0,
    hbm_gb: 192, hbm_bw_tbs: 7.4,
    nvlink_gbs: 4800, scaleout_gbs: 1200, scaleup_domain: 9216,   // largest single-pod
    power_w: 600, capex_usd: 0, is_hpc_capable: false,
    notes: "Inference-first design. Massive scale-up domain. Listed at zero capex; pricing is GCP-rental only.",
  },

  // ============================== Intel Gaudi / Falcon Shores ==============================
  GAUDI3: {
    label: "Intel Gaudi 3",
    vendor: "intel", ecosystem: "synapse", ecosystem_maturity: eco(0.65),
    availability: "merchant", status: "shipping", intro_year: 2024,
    fp16_tflops: 1835, fp8_tflops: 1835, fp32_tflops: 229, fp64_tflops: 0,
    hbm_gb: 128, hbm_bw_tbs: 3.7,
    nvlink_gbs: 1200, scaleout_gbs: 600, scaleup_domain: 8,
    power_w: 900, capex_usd: 16000, is_hpc_capable: false,
    notes: "Cheap merchant alternative. Software stack (SynapseAI) less mature; bigger MFU haircut.",
  },
  FALCON_SHORES: {
    label: "Intel Falcon Shores [announced]",
    vendor: "intel", ecosystem: "oneapi", ecosystem_maturity: eco(0.60),
    availability: "merchant", status: "announced", intro_year: 2026,
    fp16_tflops: 2200, fp8_tflops: 4400, fp32_tflops: 280, fp64_tflops: 50,
    hbm_gb: 256, hbm_bw_tbs: 9.0,
    nvlink_gbs: 1500, scaleout_gbs: 800, scaleup_domain: 8,
    power_w: 1100, capex_usd: 25000, is_hpc_capable: true,
    notes: "GPU successor to Ponte Vecchio (used in Aurora). HPC + AI hybrid.",
  },

  // ============================== Specialised / esoteric ==============================
  CEREBRAS_WSE3: {
    label: "Cerebras WSE-3 (wafer-scale)",
    vendor: "cerebras", ecosystem: "cs_studio", ecosystem_maturity: eco(0.55),
    availability: "limited", status: "shipping", intro_year: 2024,
    fp16_tflops: 125000,                   // 125 PFLOPs FP16 per system
    fp8_tflops: 250000,
    fp32_tflops: 62500, fp64_tflops: 0,
    hbm_gb: 44,                            // 44 GB on-die SRAM (no HBM)
    hbm_bw_tbs: 21000,                     // 21 PB/s on-chip; not directly comparable
    nvlink_gbs: 0, scaleout_gbs: 1200, scaleup_domain: 1,
    power_w: 23000,                        // ~23 kW per CS-3 system
    capex_usd: 2500000,                    // ~$2.5M per system
    is_hpc_capable: false,
    notes: "Wafer-scale; one chip = one node. Treat capex as 'system' price. Niche fit: large dense models, " +
           "irregular sparsity, some scientific ML. Software stack proprietary.",
  },
  GROQ_LPU: {
    label: "Groq LPU (inference-only)",
    vendor: "groq", ecosystem: "groqware", ecosystem_maturity: eco(0.50),
    availability: "limited", status: "shipping", intro_year: 2024,
    fp16_tflops: 188, fp8_tflops: 750, fp32_tflops: 188, fp64_tflops: 0,
    hbm_gb: 0,                             // 230 MB on-die SRAM
    hbm_bw_tbs: 80,                        // 80 TB/s SRAM bw
    nvlink_gbs: 0, scaleout_gbs: 100, scaleup_domain: 1,
    power_w: 215, capex_usd: 20000,
    is_hpc_capable: false,
    notes: "Deterministic execution; ultra-low latency for memory-bound LLM decode. Bad for training. SRAM-only.",
  },
  SAMBANOVA_SN40L: {
    label: "SambaNova SN40L",
    vendor: "sambanova", ecosystem: "sambaflow", ecosystem_maturity: eco(0.55),
    availability: "limited", status: "shipping", intro_year: 2024,
    fp16_tflops: 638, fp8_tflops: 1276, fp32_tflops: 159, fp64_tflops: 0,
    hbm_gb: 64, hbm_bw_tbs: 2.0,
    nvlink_gbs: 0, scaleout_gbs: 200, scaleup_domain: 8,
    power_w: 600, capex_usd: 30000,
    is_hpc_capable: false,
    notes: "Dataflow architecture. Strong for sparse / RAG / agentic workloads. Limited deployment.",
  },
};

// External rental price floor per GPU-hour (ish), used for build-vs-rent.
// Numbers are typical 1-yr reserved rates from neoclouds where applicable.
export const RENTAL_PRICE_USD_PER_GPU_HOUR = {
  A100_80GB: 1.10,
  H100_SXM: 2.00,
  H200_SXM: 2.50,
  B200: 4.00,
  GB200_NVL72: 5.00,
  R100: 7.50,
  R200_RubinUltra: 12.00,
  F100_Feynman: 20.00,
  MI300X: 1.80,
  MI325X: 2.20,
  MI355X: 3.50,
  MI300A: 3.00,        // notional; not really rentable
  TPU_v5p: 4.50,       // GCP on-demand
  TPU_v6e_Trillium: 2.70,
  TPU_v7_Ironwood: 5.00,
  GAUDI3: 1.50,
  FALCON_SHORES: 3.00,
  CEREBRAS_WSE3: 60.00,    // per-system / 1 chip; rough
  GROQ_LPU: 0.40,
  SAMBANOVA_SN40L: 2.50,
};

// Networking unit costs per GPU port (amortized BoM).
export const NETWORK_COST_PER_GPU_USD = {
  scale_up_only: 2000,
  small_pod: 8000,
  large_pod: 14000,
  full_fabric: 22000,
};

// Default datacenter constants
export const DC = {
  pue: 1.25,
  electricity_usd_per_kwh: 0.07,
  facility_capex_per_kw_usd: 9000,
  facility_opex_pct_of_capex: 0.04,
};

// Status-tier helpers for UI grouping/labeling.
export const STATUS_LABELS = {
  shipping: "Shipping today",
  announced: "Announced (near-term)",
  speculative: "Speculative roadmap",
};

export const VENDOR_LABELS = {
  nvidia: "NVIDIA",
  amd: "AMD",
  google: "Google",
  intel: "Intel",
  cerebras: "Cerebras",
  groq: "Groq",
  sambanova: "SambaNova",
};

export const AVAILABILITY_LABELS = {
  merchant: "Merchant (open market)",
  gcp_only: "GCP rental only",
  doe_only: "DOE-procured (not generally available)",
  limited: "Limited availability / specialised channel",
  speculative: "Vendor roadmap (not yet shipping)",
};
