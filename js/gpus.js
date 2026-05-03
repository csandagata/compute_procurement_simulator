// GPU reference database.
// Numbers are public/published spec sheets and reasonable street-price proxies as of late 2025.
// FLOP figures are dense unless noted. NVLink is per-GPU bidirectional.
//
// FP64 figures are TF64 / Tensor Core where applicable. FP64 matters for HPC
// workloads (CFD, fusion plasma, materials DFT, particle event reconstruction)
// — note B200 actually downgraded FP64 vs H100 to optimise for AI; MI300X is
// best-in-class for FP64.

export const GPUS = {
  A100_80GB: {
    label: "NVIDIA A100 80GB SXM",
    fp16_tflops: 312,        // dense FP16 / BF16
    fp8_tflops: 0,           // no FP8 path
    fp32_tflops: 19.5,       // TF32 dense; non-tensor FP32 ~19.5 too
    fp64_tflops: 19.5,       // TF64 tensor core
    hbm_gb: 80,
    hbm_bw_tbs: 2.0,
    nvlink_gbs: 600,         // intra-node
    scaleout_gbs: 200,       // typical IB HDR per GPU
    power_w: 400,
    capex_usd: 12000,        // secondary-market typical
    intro_year: 2020,
    scaleup_domain: 8,       // DGX A100 NVLink switch fabric
    is_hpc_capable: true,
    vendor: "nvidia",
  },
  H100_SXM: {
    label: "NVIDIA H100 SXM 80GB",
    fp16_tflops: 989,
    fp8_tflops: 1979,
    fp32_tflops: 67,
    fp64_tflops: 67,
    hbm_gb: 80,
    hbm_bw_tbs: 3.35,
    nvlink_gbs: 900,
    scaleout_gbs: 400,       // CX-7 NDR
    power_w: 700,
    capex_usd: 28000,
    intro_year: 2023,
    scaleup_domain: 8,
    is_hpc_capable: true,
    vendor: "nvidia",
  },
  H200_SXM: {
    label: "NVIDIA H200 SXM 141GB",
    fp16_tflops: 989,
    fp8_tflops: 1979,
    fp32_tflops: 67,
    fp64_tflops: 67,
    hbm_gb: 141,
    hbm_bw_tbs: 4.8,
    nvlink_gbs: 900,
    scaleout_gbs: 400,
    power_w: 700,
    capex_usd: 32000,
    intro_year: 2024,
    scaleup_domain: 8,
    is_hpc_capable: true,
    vendor: "nvidia",
  },
  B200: {
    label: "NVIDIA B200 SXM",
    fp16_tflops: 2250,
    fp8_tflops: 4500,
    fp32_tflops: 80,
    fp64_tflops: 37,         // intentional downgrade vs H100; B200 prioritises AI
    hbm_gb: 192,
    hbm_bw_tbs: 8.0,
    nvlink_gbs: 1800,
    scaleout_gbs: 800,       // CX-8
    power_w: 1000,
    capex_usd: 40000,
    intro_year: 2025,
    scaleup_domain: 8,
    is_hpc_capable: false,   // poor FP64; not a great fit for CFD/fusion
    vendor: "nvidia",
  },
  GB200_NVL72: {
    label: "NVIDIA GB200 (NVL72 rack-scale)",
    fp16_tflops: 2500,
    fp8_tflops: 5000,
    fp32_tflops: 90,
    fp64_tflops: 45,
    hbm_gb: 192,
    hbm_bw_tbs: 8.0,
    nvlink_gbs: 1800,        // intra-rack scale-up domain of 72 GPUs
    scaleout_gbs: 800,
    power_w: 1200,
    capex_usd: 50000,
    intro_year: 2025,
    scaleup_domain: 72,
    is_hpc_capable: false,
    vendor: "nvidia",
  },
  MI300X: {
    label: "AMD MI300X 192GB",
    fp16_tflops: 1300,
    fp8_tflops: 2600,
    fp32_tflops: 163,        // matrix FP32
    fp64_tflops: 81,         // best-in-class FP64 matrix
    hbm_gb: 192,
    hbm_bw_tbs: 5.3,
    nvlink_gbs: 896,         // Infinity Fabric
    scaleout_gbs: 400,
    power_w: 750,
    capex_usd: 18000,
    intro_year: 2024,
    scaleup_domain: 8,
    is_hpc_capable: true,
    vendor: "amd",
  },
};

// External rental price floor per GPU-hour, used for build-vs-rent comparison.
// Typical neocloud reserved 1-yr rates.
export const RENTAL_PRICE_USD_PER_GPU_HOUR = {
  A100_80GB: 1.10,
  H100_SXM: 2.00,
  H200_SXM: 2.50,
  B200: 4.00,
  GB200_NVL72: 5.00,
  MI300X: 1.80,
};

// Networking unit costs per GPU port, including switches, transceivers, cables.
// Figures are amortized BoM estimates for IB NDR/XDR class fabrics.
export const NETWORK_COST_PER_GPU_USD = {
  scale_up_only: 2000,      // single-rack workloads
  small_pod: 8000,          // 1-2 SU, IB or NVL72
  large_pod: 14000,         // multi-SU rail-optimized
  full_fabric: 22000,       // datacenter-spanning fat-tree
};

// Default datacenter constants
export const DC = {
  pue: 1.25,
  electricity_usd_per_kwh: 0.07,
  facility_capex_per_kw_usd: 9000,   // shell + power + cooling, per IT-kW
  facility_opex_pct_of_capex: 0.04,  // per year
};
