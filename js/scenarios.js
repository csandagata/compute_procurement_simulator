// Preset scenarios. Each is a complete input set for buildPlan().
// Loading a preset overwrites all controls.
//
// Two flavors:
//   - legacy scenarios use {interactiveTokensPerDay, ...} inputs and leave
//     `portfolio` empty; buildPlan synthesizes a portfolio from them.
//   - new scenarios specify `portfolio` explicitly using makePortfolioItem
//     and exercise the AI-for-science workload classes.

import { makePortfolioItem } from "./portfolio.js";

const COMMON_DEFAULTS = {
  outputLen: 200,
  kvScheme: "GQA_8",
  kvSchemeOverride: false,
  mlaCompressionDim: 576,
  enableSpecDec: true,
  specDecAcceptanceProb: 0.7,
  specDecGammaMax: 4,
  specDecDraftCost: 0.08,
  enableQueueing: true,
  allowedGpus: null,           // null = all SKUs allowed
  maxTpPerWorkload: null,      // null = use scaleup_domain × 4
  inputOutputRatio: null,      // null = use contextLen/outputLen
};

export const SCENARIOS = {
  enterprise_balanced: {
    label: "Enterprise: balanced AI org",
    description:
      "Mid-size enterprise running an internal copilot for 50k employees, " +
      "modest training program, growing 60% YoY. CUDA-only fleet (no AMD).",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      gpuKey: "H100_SXM",
      quantization: "fp8",
      paramsB: 70,
      activeFrac: 1.0,
      contextLen: 8000,
      outputLen: 250,
      kvBytesPerToken: 200_000,
      kvScheme: "GQA_8",
      interactiveTokensPerDay: 5e9,
      batchTokensPerDay: 20e9,
      rlTokensPerDay: 0,
      pretrainFlopsPerYear: 1e23,
      finetuneFlopsPerYear: 5e22,
      tpDegree: 8,
      mfu: 0.35,
      utilization: 0.6,
      demandGrowth: 0.6,
      modelGrowth: 0.4,
      algEfficiency: 0.5,
      refreshYears: 4,
      discountRate: 0.10,
      externalInitialPrice: 5.0,
      externalDeclineMult: 3.0,
      edgeFacilityShare: 0.2,
      regionalFacilityShare: 0.4,
      centralFacilityShare: 0.4,
      facilityMaxMw: 30,
      headroomFactor: 1.25,
      rentalShare: 0.2,
      pue: 1.25,
      electricityPrice: 0.07,
      networkTier: "small_pod",
      allowedGpus: ["H100_SXM", "H200_SXM", "B200"],
    },
  },
  frontier_lab: {
    label: "Frontier lab: training-heavy, fast scaling",
    description:
      "AI lab pushing the frontier, large pretrain compute, RL post-training, " +
      "model size doubling each year. Open to all hardware.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      gpuKey: "GB200_NVL72",
      quantization: "fp8",
      paramsB: 500,
      activeFrac: 0.1,
      contextLen: 32000,
      outputLen: 500,
      kvBytesPerToken: 60_000,
      kvScheme: "MLA",
      mlaCompressionDim: 576,
      interactiveTokensPerDay: 50e9,
      batchTokensPerDay: 200e9,
      rlTokensPerDay: 100e9,
      pretrainFlopsPerYear: 5e25,
      finetuneFlopsPerYear: 5e24,
      tpDegree: 16,
      mfu: 0.42,
      utilization: 0.7,
      demandGrowth: 1.5,
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
      electricityPrice: 0.05,
      networkTier: "full_fabric",
      specDecAcceptanceProb: 0.8,
      specDecGammaMax: 5,
    },
  },
  inference_only: {
    label: "Inference-only product company",
    description:
      "Company serves a global consumer product with strict <30ms/tok latency. " +
      "No training; uses open-weight models. Heterogeneous fleet for batch vs interactive.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      gpuKey: "H200_SXM",
      quantization: "fp8",
      paramsB: 70,
      activeFrac: 1.0,
      contextLen: 4000,
      outputLen: 300,
      kvBytesPerToken: 200_000,
      kvScheme: "GQA_8",
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
      edgeFacilityShare: 0.5,
      regionalFacilityShare: 0.3,
      centralFacilityShare: 0.2,
      facilityMaxMw: 20,
      headroomFactor: 1.4,
      rentalShare: 0.3,
      pue: 1.3,
      electricityPrice: 0.10,
      networkTier: "small_pod",
      allowedGpus: ["H100_SXM", "H200_SXM", "B200", "MI300X"],
    },
  },
  conservative: {
    label: "Conservative: cloud-first, hedged",
    description:
      "Risk-averse enterprise. Heavy cloud rental, only owns capacity for " +
      "baseload, large headroom, slow refresh. NVIDIA-only.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      gpuKey: "H100_SXM",
      quantization: "fp16",
      paramsB: 30,
      activeFrac: 1.0,
      contextLen: 8000,
      outputLen: 200,
      kvBytesPerToken: 100_000,
      kvScheme: "GQA_8",
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
      allowedGpus: ["A100_80GB", "H100_SXM"],
      enableSpecDec: false,
    },
  },

  // -------------------- AI-for-science / digital-twin scenarios --------------------

  pharma_rd: {
    label: "Pharma R&D: AI-for-biology lab",
    description:
      "Frontier scaling outsourced. In-house compute for protein structure prediction, " +
      "drug docking, multi-omics, plus modest internal LLM serving. Heterogeneous fleet.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      paramsB: 70, activeFrac: 1.0, quantization: "fp8",
      contextLen: 8000, outputLen: 250, kvScheme: "GQA_8",
      // No legacy demand fields — portfolio specifies workload mix.
      mfu: 0.30, utilization: 0.65,
      demandGrowth: 0, modelGrowth: 0,    // not used; per-item growth in portfolio
      algEfficiency: 0.4,
      refreshYears: 4, discountRate: 0.10,
      externalInitialPrice: 3.0, externalDeclineMult: 3.0,
      edgeFacilityShare: 0.1, regionalFacilityShare: 0.3, centralFacilityShare: 0.6,
      facilityMaxMw: 30, headroomFactor: 1.3,
      rentalShare: 0.15, pue: 1.25, electricityPrice: 0.07,
      networkTier: "small_pod",
      allowedGpus: ["H100_SXM", "H200_SXM", "B200", "MI300X"],
      portfolio: [
        makePortfolioItem("alphafold_screening",     { unitsPerYear: 5e8, scalingCase: "median" }),
        makePortfolioItem("protein_design_rl",       { unitsPerYear: 1e7, scalingCase: "median" }),
        makePortfolioItem("drug_docking",            { unitsPerYear: 5e10, scalingCase: "median" }),
        makePortfolioItem("multi_omics_integration", { unitsPerYear: 1e7, scalingCase: "low" }),  // pessimistic — data-bound
        makePortfolioItem("ml_force_field_md",       { unitsPerYear: 1e6, scalingCase: "median" }),
        makePortfolioItem("llm_interactive",         { unitsPerYear: 1e9 * 365, scalingCase: "median" }),
        makePortfolioItem("llm_finetune",            { unitsPerYear: 1e22, scalingCase: "median" }),
      ],
    },
  },

  climate_energy_lab: {
    label: "Climate / energy national lab: FP64 + ML hybrid",
    description:
      "FP64-bound classical sims (CFD, fusion plasma, climate) + ML weather models. " +
      "Filter to HPC-capable SKUs (H100, MI300X) — B200 is a poor fit.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      paramsB: 50, activeFrac: 1.0, quantization: "fp8",
      contextLen: 4000, outputLen: 200, kvScheme: "GQA_8",
      mfu: 0.45, utilization: 0.85,
      demandGrowth: 0, modelGrowth: 0, algEfficiency: 0.3,
      refreshYears: 5, discountRate: 0.08,
      externalInitialPrice: 4.0, externalDeclineMult: 2.5,
      edgeFacilityShare: 0.0, regionalFacilityShare: 0.2, centralFacilityShare: 0.8,
      facilityMaxMw: 100, headroomFactor: 1.2,
      rentalShare: 0.0, pue: 1.18, electricityPrice: 0.05,
      networkTier: "large_pod",
      allowedGpus: ["H100_SXM", "H200_SXM", "MI300X"],   // HPC-capable only
      portfolio: [
        makePortfolioItem("cfd_digital_twin",      { unitsPerYear: 1e6, scalingCase: "median" }),
        makePortfolioItem("fusion_plasma_sim",     { unitsPerYear: 5e3, scalingCase: "high" }),  // optimistic: ML surrogates unlock
        makePortfolioItem("climate_classical",     { unitsPerYear: 50,  scalingCase: "median" }),
        makePortfolioItem("weather_graphnet",      { unitsPerYear: 1e5, scalingCase: "median" }),
        makePortfolioItem("astrophysics_sim",      { unitsPerYear: 1e3, scalingCase: "median" }),
        makePortfolioItem("llm_batch",             { unitsPerYear: 5e9 * 365, scalingCase: "median" }),
      ],
    },
  },

  industrial_research: {
    label: "Industrial research consortium: digital twins + materials",
    description:
      "Manufacturing digital twins, materials discovery campaigns, network/grid " +
      "simulation. FP64-heavy with ML acceleration; modest LLM load.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      paramsB: 30, activeFrac: 1.0, quantization: "fp16",
      contextLen: 4000, outputLen: 200, kvScheme: "GQA_8",
      mfu: 0.40, utilization: 0.7,
      demandGrowth: 0, modelGrowth: 0, algEfficiency: 0.3,
      refreshYears: 4, discountRate: 0.10,
      externalInitialPrice: 4.0, externalDeclineMult: 2.5,
      edgeFacilityShare: 0.1, regionalFacilityShare: 0.4, centralFacilityShare: 0.5,
      facilityMaxMw: 50, headroomFactor: 1.3,
      rentalShare: 0.10, pue: 1.25, electricityPrice: 0.07,
      networkTier: "small_pod",
      allowedGpus: ["H100_SXM", "H200_SXM", "MI300X"],
      portfolio: [
        makePortfolioItem("manufacturing_digital_twin", { unitsPerYear: 1e4, scalingCase: "median" }),
        makePortfolioItem("materials_active_learning",  { unitsPerYear: 5e5, scalingCase: "high" }),
        makePortfolioItem("ml_force_field_md",          { unitsPerYear: 5e5, scalingCase: "median" }),
        makePortfolioItem("network_digital_twin",       { unitsPerYear: 1e5, scalingCase: "median" }),
        makePortfolioItem("llm_interactive",            { unitsPerYear: 5e8 * 365, scalingCase: "low" }),
        makePortfolioItem("llm_finetune",               { unitsPerYear: 1e22, scalingCase: "low" }),
      ],
    },
  },

  defense_intel: {
    label: "Defense / intelligence: imagery + reconstruction + LLM",
    description:
      "Satellite imagery at scale, particle-event-style reconstruction for sensor data, " +
      "LLM for analyst workflows, with sovereignty constraints on placement.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      paramsB: 70, activeFrac: 1.0, quantization: "fp8",
      contextLen: 8000, outputLen: 200, kvScheme: "GQA_8",
      mfu: 0.30, utilization: 0.6,
      demandGrowth: 0, modelGrowth: 0, algEfficiency: 0.5,
      refreshYears: 4, discountRate: 0.08,
      externalInitialPrice: 5.0, externalDeclineMult: 2.0,
      edgeFacilityShare: 0.4, regionalFacilityShare: 0.4, centralFacilityShare: 0.2,
      facilityMaxMw: 30, headroomFactor: 1.5,
      rentalShare: 0.0, pue: 1.30, electricityPrice: 0.10,    // cleared facilities
      networkTier: "small_pod",
      allowedGpus: ["H100_SXM", "H200_SXM", "B200"],          // CUDA-only, vetted
      portfolio: [
        makePortfolioItem("satellite_imagery",       { unitsPerYear: 5e8, scalingCase: "high" }),
        makePortfolioItem("particle_event_reco",     { unitsPerYear: 1e11, scalingCase: "median" }),
        makePortfolioItem("llm_interactive",         { unitsPerYear: 2e9 * 365, scalingCase: "median" }),
        makePortfolioItem("llm_batch",               { unitsPerYear: 5e9 * 365, scalingCase: "median" }),
      ],
    },
  },

  speculative_ai_for_science: {
    label: "Speculative: compute-bound AI-for-science (high case)",
    description:
      "What if every AI-for-science domain turns out to be compute-bound? " +
      "Same portfolio as pharma+climate, all dialed to high scaling case. Stress-test of fleet & power.",
    inputs: {
      ...COMMON_DEFAULTS,
      horizonYears: 5,
      paramsB: 70, activeFrac: 1.0, quantization: "fp8",
      contextLen: 4000, outputLen: 200, kvScheme: "GQA_8",
      mfu: 0.40, utilization: 0.7,
      demandGrowth: 0, modelGrowth: 0, algEfficiency: 0.5,
      refreshYears: 3, discountRate: 0.10,
      externalInitialPrice: 5.0, externalDeclineMult: 3.0,
      edgeFacilityShare: 0.1, regionalFacilityShare: 0.3, centralFacilityShare: 0.6,
      facilityMaxMw: 100, headroomFactor: 1.3,
      rentalShare: 0.10, pue: 1.20, electricityPrice: 0.05,
      networkTier: "large_pod",
      allowedGpus: ["H100_SXM", "H200_SXM", "B200", "MI300X"],
      portfolio: [
        makePortfolioItem("alphafold_screening",     { unitsPerYear: 5e8, scalingCase: "high" }),
        makePortfolioItem("protein_design_rl",       { unitsPerYear: 1e7, scalingCase: "high" }),
        makePortfolioItem("multi_omics_integration", { unitsPerYear: 1e7, scalingCase: "high" }),
        makePortfolioItem("fusion_plasma_sim",       { unitsPerYear: 5e3, scalingCase: "high" }),
        makePortfolioItem("weather_graphnet",        { unitsPerYear: 1e5, scalingCase: "high" }),
        makePortfolioItem("satellite_imagery",       { unitsPerYear: 5e8, scalingCase: "high" }),
        makePortfolioItem("materials_active_learning",{ unitsPerYear: 5e5, scalingCase: "high" }),
      ],
    },
  },
};

export const DEFAULT_SCENARIO = "enterprise_balanced";
