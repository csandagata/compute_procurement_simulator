// Hats / lenses — the dashboard adapts emphasis to the user's role.
//
// Five hats are first-class. Each declares:
//   - which frames are primary (highlighted)
//   - which charts and KPIs are most relevant
//   - a one-line "what to focus on" blurb
//   - hat-specific framing for the narrative panel
//
// "all" is the default — show everything equally without re-emphasis.

export const HATS = {
  all: {
    id: "all",
    label: "All views",
    short: "Everyone",
    icon: "◉",
    blurb: "Show the full dashboard with no emphasis.",
    primaryFrames: [],
    primaryCharts: [],
    keyKpis: ["kpi-peak-gpus", "kpi-peak-mw", "kpi-peak-facilities", "kpi-tco", "kpi-internal-price", "kpi-fabric"],
  },

  planner: {
    id: "planner",
    label: "Planning strategist",
    short: "Strategist",
    icon: "▦",
    blurb:
      "You're sizing the physical buildout: how many chips of which kind across how " +
      "many facilities, drawing how much power, costing how much over five years. " +
      "Lead times and refresh cycles drive your decisions.",
    primaryFrames: ["frame-chip", "frame-latency"],
    primaryCharts: ["chart-sku-mix", "chart-tco", "chart-facilities", "chart-power", "chart-buybuild"],
    keyKpis: ["kpi-peak-gpus", "kpi-peak-mw", "kpi-peak-facilities", "kpi-tco"],
    questions: [
      "How many GPUs in total across the horizon, broken down by SKU and refresh year?",
      "How many distinct facilities and how many MW each?",
      "When do we hit the announced/speculative chip generations and what's the buy-now-vs-wait math?",
    ],
  },

  workload: {
    id: "workload",
    label: "Product / workload strategist",
    short: "Workload",
    icon: "◇",
    blurb:
      "You compose the application portfolio — chat, batch, training, RL, AI-for-science. " +
      "You care about how demand mix and per-workload growth shapes the fleet.",
    primaryFrames: ["frame-portfolio"],
    primaryCharts: ["chart-domain", "chart-fp64-split", "chart-bound-heatmap", "chart-scaling-fan"],
    keyKpis: ["kpi-peak-gpus", "kpi-internal-price"],
    questions: [
      "What domain dominates the GPU-year pool, and is that what we intended?",
      "Where does the bottleneck live per workload — memory, compute, network, FP64?",
      "How wide is the latent-scaling-law uncertainty band for the non-LLM domains?",
    ],
  },

  finance: {
    id: "finance",
    label: "Financial planner",
    short: "Finance",
    icon: "$",
    blurb:
      "You weigh project NPV, demand uncertainty, and optionality. Your output is a " +
      "go/no-go on capital deployment given the spread of outcomes — not the median.",
    primaryFrames: ["frame-chip", "frame-longterm"],
    primaryCharts: ["chart-tco", "chart-buybuild", "chart-fan", "chart-tornado", "chart-scaling-fan"],
    keyKpis: ["kpi-tco", "kpi-internal-price"],
    questions: [
      "What's the TCO sensitivity to the top three input levers?",
      "Across demand cases, what's the P10/P90 spread in fleet need?",
      "Where does internal cost cross external-rental cost — how robust is that crossing?",
    ],
  },

  ai_engineer: {
    id: "ai_engineer",
    label: "AI / ML platform engineer",
    short: "AI Eng",
    icon: "λ",
    blurb:
      "You're forecasting algorithmic and architectural shifts: KV schemes, spec-dec, " +
      "quantization, parallelism. Your outputs feed throughput per GPU and what the next chip should target.",
    primaryFrames: ["frame-portfolio", "frame-latency"],
    primaryCharts: ["chart-bound-heatmap", "chart-latency", "chart-scaleup", "chart-fp64-split"],
    keyKpis: ["kpi-internal-price", "kpi-fabric"],
    questions: [
      "Which workloads are memory-bw-bound vs compute-bound vs network-bound today, and how does that flip with FP4 / spec-dec / MLA?",
      "Does the current fabric size accommodate next year's frontier model, or do we need to re-architect parallelism?",
      "What's the latency-cost Pareto frontier if our quantization assumptions are wrong?",
    ],
  },

  ops: {
    id: "ops",
    label: "Operations / facilities",
    short: "Ops",
    icon: "⚙",
    blurb:
      "You run what was bought: power, cooling, facilities, networking. Your concern " +
      "is MW available, PUE, electricity cost, and the refresh schedule disrupting steady-state ops.",
    primaryFrames: ["frame-latency", "frame-longterm"],
    primaryCharts: ["chart-power", "chart-facilities", "chart-scaleup", "chart-fan"],
    keyKpis: ["kpi-peak-mw", "kpi-peak-facilities", "kpi-fabric"],
    questions: [
      "What's the peak MW load and across how many facilities?",
      "When do we need to power on new capacity given 2–3 year lead times?",
      "What's the year-over-year refresh rhythm, and does it match our depreciation plan?",
    ],
  },
};

export const DEFAULT_HAT = "all";

// Map frame ids to their human labels (used by the lens highlighter).
export const FRAME_LABELS = {
  "frame-portfolio": "Workload portfolio",
  "frame-chip":      "Chip mix & spend",
  "frame-latency":   "Latency & data center needs",
  "frame-longterm":  "Long-term & surge implications",
};

// KPI metadata: didactic explanations + which hats care about each.
// The UI renders an info icon that surfaces this on hover.
export const KPI_META = {
  "kpi-peak-gpus": {
    label: "Peak fleet",
    explainer:
      "Maximum GPU count across the planning horizon, summed over all SKUs. " +
      "This is what you need to physically procure (plus the headroom factor).",
    hats: ["planner", "workload", "ops"],
  },
  "kpi-peak-mw": {
    label: "Peak power",
    explainer:
      "Maximum facility power draw including PUE overhead. Drives site selection " +
      "and grid-interconnect lead times (typically 24–36 months for new MW).",
    hats: ["planner", "ops"],
  },
  "kpi-peak-facilities": {
    label: "Facilities",
    explainer:
      "Number of distinct facilities the fleet is split across, given latency-tier " +
      "policy and the per-facility max-MW constraint.",
    hats: ["planner", "ops"],
  },
  "kpi-tco": {
    label: "Discounted TCO",
    explainer:
      "Net present value of all CapEx + OpEx over the horizon, discounted at the " +
      "specified rate. Includes GPUs, networking, facility build, power, and rentals.",
    hats: ["planner", "finance"],
  },
  "kpi-internal-price": {
    label: "Internal $/Mtok (final yr)",
    explainer:
      "Cost per million LLM-inference tokens in the final year, allocated by " +
      "inference's share of total GPU-years (so training-heavy scenarios don't " +
      "double-count pretraining CapEx). Compare against external rental benchmark.",
    hats: ["finance", "workload", "ai_engineer"],
  },
  "kpi-fabric": {
    label: "Coherent fabric needed",
    explainer:
      "Smallest scale-up domain (NVLink/IB) required to host the year's frontier " +
      "model after quantization. If this exceeds the chosen SKU's scaleup_domain, " +
      "you need pipeline parallelism across pods or different hardware.",
    hats: ["ai_engineer", "planner"],
  },
};
