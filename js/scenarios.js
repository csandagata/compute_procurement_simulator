// Preset scenarios. Each is a complete input set for buildPlan().
// Loading a preset overwrites all controls.

export const SCENARIOS = {
  enterprise_balanced: {
    label: "Enterprise: balanced AI org",
    description:
      "Mid-size enterprise running an internal copilot for 50k employees, " +
      "modest training program, growing 60% YoY.",
    inputs: {
      horizonYears: 5,
      gpuKey: "H100_SXM",
      quantization: "fp8",
      paramsB: 70,
      activeFrac: 1.0,
      contextLen: 8000,
      kvBytesPerToken: 200_000,    // 200 KB per token typical for 70B at FP16 KV
      interactiveTokensPerDay: 5e9,
      batchTokensPerDay: 20e9,
      rlTokensPerDay: 0,
      pretrainFlopsPerYear: 1e23,  // ~one mid-size pretrain run
      finetuneFlopsPerYear: 5e22,
      tpDegree: 8,
      mfu: 0.35,
      utilization: 0.6,
      demandGrowth: 0.6,
      modelGrowth: 0.4,
      algEfficiency: 0.5,
      refreshYears: 4,
      discountRate: 0.10,
      externalInitialPrice: 5.0,    // $/Mtok at the capability tier today
      externalDeclineMult: 3.0,     // 3x cheaper per year baseline
      edgeFacilityShare: 0.2,
      regionalFacilityShare: 0.4,
      centralFacilityShare: 0.4,
      facilityMaxMw: 30,
      headroomFactor: 1.25,
      rentalShare: 0.2,
      pue: 1.25,
      electricityPrice: 0.07,
      networkTier: "small_pod",
    },
  },
  frontier_lab: {
    label: "Frontier lab: training-heavy, fast scaling",
    description:
      "AI lab pushing the frontier, large pretrain compute, RL post-training, " +
      "model size doubling each year.",
    inputs: {
      horizonYears: 5,
      gpuKey: "GB200_NVL72",
      quantization: "fp8",
      paramsB: 500,
      activeFrac: 0.1,             // MoE sparsity ~10x
      contextLen: 32000,
      kvBytesPerToken: 60_000,     // MLA / GQA compressed
      interactiveTokensPerDay: 50e9,
      batchTokensPerDay: 200e9,
      rlTokensPerDay: 100e9,
      pretrainFlopsPerYear: 5e25,
      finetuneFlopsPerYear: 5e24,
      tpDegree: 16,
      mfu: 0.42,
      utilization: 0.7,
      demandGrowth: 1.5,           // 2.5x per year
      modelGrowth: 0.7,
      algEfficiency: 0.8,
      refreshYears: 3,
      discountRate: 0.12,
      externalInitialPrice: 15.0,
      externalDeclineMult: 4.0,
      edgeFacilityShare: 0.1,
      regionalFacilityShare: 0.2,
      centralFacilityShare: 0.7,
      facilityMaxMw: 150,
      headroomFactor: 1.3,
      rentalShare: 0.05,
      pue: 1.20,
      electricityPrice: 0.05,      // power-cost-optimized siting
      networkTier: "full_fabric",
    },
  },
  inference_only: {
    label: "Inference-only product company",
    description:
      "Company serves a global consumer product with strict <30ms/tok latency. " +
      "No training; uses open-weight models.",
    inputs: {
      horizonYears: 5,
      gpuKey: "H200_SXM",
      quantization: "fp8",
      paramsB: 70,
      activeFrac: 1.0,
      contextLen: 4000,
      kvBytesPerToken: 200_000,
      interactiveTokensPerDay: 100e9,
      batchTokensPerDay: 20e9,
      rlTokensPerDay: 0,
      pretrainFlopsPerYear: 0,
      finetuneFlopsPerYear: 1e22,
      tpDegree: 8,
      mfu: 0.3,
      utilization: 0.55,
      demandGrowth: 0.8,
      modelGrowth: 0.2,
      algEfficiency: 0.6,
      refreshYears: 4,
      discountRate: 0.10,
      externalInitialPrice: 2.0,
      externalDeclineMult: 3.0,
      edgeFacilityShare: 0.5,       // latency drives geography
      regionalFacilityShare: 0.3,
      centralFacilityShare: 0.2,
      facilityMaxMw: 20,
      headroomFactor: 1.4,
      rentalShare: 0.3,
      pue: 1.3,
      electricityPrice: 0.10,
      networkTier: "small_pod",
    },
  },
  conservative: {
    label: "Conservative: cloud-first, hedged",
    description:
      "Risk-averse enterprise. Heavy cloud rental, only owns capacity for " +
      "baseload, large headroom, slow refresh.",
    inputs: {
      horizonYears: 5,
      gpuKey: "H100_SXM",
      quantization: "fp16",
      paramsB: 30,
      activeFrac: 1.0,
      contextLen: 8000,
      kvBytesPerToken: 100_000,
      interactiveTokensPerDay: 1e9,
      batchTokensPerDay: 3e9,
      rlTokensPerDay: 0,
      pretrainFlopsPerYear: 0,
      finetuneFlopsPerYear: 1e22,
      tpDegree: 4,
      mfu: 0.3,
      utilization: 0.5,
      demandGrowth: 0.3,
      modelGrowth: 0.1,
      algEfficiency: 0.4,
      refreshYears: 5,
      discountRate: 0.08,
      externalInitialPrice: 4.0,
      externalDeclineMult: 2.5,
      edgeFacilityShare: 0.0,
      regionalFacilityShare: 0.4,
      centralFacilityShare: 0.6,
      facilityMaxMw: 15,
      headroomFactor: 1.5,
      rentalShare: 0.6,
      pue: 1.4,
      electricityPrice: 0.10,
      networkTier: "scale_up_only",
    },
  },
};

export const DEFAULT_SCENARIO = "enterprise_balanced";
