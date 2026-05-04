// Plotly chart helpers. Each function takes a plan object and a target div id.

import { DOMAINS, WORKLOAD_CLASSES } from "./workload_classes.js";

const COLORS = {
  interactive_inference: "#4F8EF7",
  batch_inference: "#7BB661",
  rl: "#E08E45",
  training: "#C8553D",
  finetune: "#9F86C0",
  edge: "#5BC0BE",
  regional: "#3A506B",
  central: "#1C2541",
  capex_gpu: "#4F8EF7",
  capex_network: "#7BB661",
  capex_facility: "#E08E45",
  opex_electricity: "#C8553D",
  opex_facility: "#9F86C0",
  opex_rental: "#A0A0A0",
  internal: "#1C2541",
  external: "#C8553D",
  fp64: "#C8553D",
  non_fp64: "#4F8EF7",
};

// Distinct color palette for SKUs (used by the chip-mix chart).
const SKU_COLORS = {
  A100_80GB:    "#76B041",
  H100_SXM:     "#76A1F2",
  H200_SXM:     "#3B72D9",
  B200:         "#1C2541",
  GB200_NVL72:  "#0F4C81",
  MI300X:       "#E08E45",
};

// Bound-type colors used by both the stacked-bar and heatmap.
const BOUND_COLORS = {
  memory:  "#4F8EF7",
  compute: "#7BB661",
  network: "#E08E45",
  fp64:    "#C8553D",
  none:    "#444",
};

const LAYOUT_BASE = {
  paper_bgcolor: "#0f1117",
  plot_bgcolor: "#0f1117",
  font: { color: "#e6e6e6", family: "Inter, system-ui, sans-serif", size: 12 },
  margin: { l: 60, r: 30, t: 50, b: 50 },
  legend: { bgcolor: "rgba(0,0,0,0)" },
  hoverlabel: { bgcolor: "#1a1d29", font: { color: "#fff" } },
};

function fmt(n) {
  if (!isFinite(n)) return "—";
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return n.toFixed(0);
}

export function renderFleetByWorkload(divId, plan) {
  const years = plan.yearly.map((y) => `Y${y.year}`);
  const traces = [];
  for (const wl of ["interactive_inference", "batch_inference", "rl", "training", "finetune"]) {
    traces.push({
      type: "bar",
      x: years,
      y: plan.yearly.map((y) => y.gpuYears[wl]),
      name: wl.replace("_", " "),
      marker: { color: COLORS[wl] },
      hovertemplate: "%{y:.0f} GPU-years<extra>%{fullData.name}</extra>",
    });
  }
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    barmode: "stack",
    title: "GPU-years required by workload",
    xaxis: { title: "Year" },
    yaxis: { title: "GPU-years" },
  }, { displaylogo: false, responsive: true });
}

export function renderFacilities(divId, plan) {
  const years = plan.yearly.map((y) => `Y${y.year}`);
  const traces = [];
  for (const tier of ["edge", "regional", "central"]) {
    traces.push({
      type: "bar",
      x: years,
      y: plan.yearly.map((y) => y.tierFacilityCount[tier]),
      name: tier,
      marker: { color: COLORS[tier] },
      hovertemplate: "%{y} facilities<extra>%{fullData.name} tier</extra>",
    });
  }
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    barmode: "stack",
    title: "Facility count by latency tier",
    xaxis: { title: "Year" },
    yaxis: { title: "Facilities", dtick: 1 },
  }, { displaylogo: false, responsive: true });
}

export function renderPower(divId, plan) {
  const years = plan.yearly.map((y) => y.year);
  const traces = [{
    type: "scatter",
    mode: "lines+markers",
    x: years,
    y: plan.yearly.map((y) => y.totalKw / 1000),
    line: { color: "#4F8EF7", width: 3 },
    marker: { size: 8 },
    name: "Total IT+overhead MW",
    hovertemplate: "Year %{x}: %{y:.1f} MW<extra></extra>",
  }];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    title: "Power footprint (MW, includes PUE overhead)",
    xaxis: { title: "Year", dtick: 1 },
    yaxis: { title: "MW" },
  }, { displaylogo: false, responsive: true });
}

export function renderTcoStack(divId, plan) {
  const years = plan.yearly.map((y) => `Y${y.year}`);
  const traces = [
    { type: "bar", x: years, y: plan.yearly.map((y) => y.capex.gpu),       name: "CapEx: GPUs",      marker: { color: COLORS.capex_gpu } },
    { type: "bar", x: years, y: plan.yearly.map((y) => y.capex.network),   name: "CapEx: Networking", marker: { color: COLORS.capex_network } },
    { type: "bar", x: years, y: plan.yearly.map((y) => y.capex.facility),  name: "CapEx: Facility",  marker: { color: COLORS.capex_facility } },
    { type: "bar", x: years, y: plan.yearly.map((y) => y.opex.electricity),name: "OpEx: Power",      marker: { color: COLORS.opex_electricity } },
    { type: "bar", x: years, y: plan.yearly.map((y) => y.opex.facility),   name: "OpEx: Facility",   marker: { color: COLORS.opex_facility } },
    { type: "bar", x: years, y: plan.yearly.map((y) => y.opex.rental),     name: "OpEx: Rental",     marker: { color: COLORS.opex_rental } },
  ];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    barmode: "stack",
    title: "TCO breakdown by year (USD)",
    xaxis: { title: "Year" },
    yaxis: { title: "USD/year", tickformat: "$.2s" },
  }, { displaylogo: false, responsive: true });
}

export function renderBuildVsBuy(divId, plan) {
  const years = plan.yearly.map((y) => y.year);
  const traces = [
    {
      type: "scatter", mode: "lines+markers", x: years,
      y: plan.yearly.map((y) => y.internalDollarsPerMtok),
      name: "Internal $/Mtok (your fleet)", line: { color: COLORS.internal, width: 3 },
    },
    {
      type: "scatter", mode: "lines+markers", x: years,
      y: plan.yearly.map((y) => y.externalDollarsPerMtok),
      name: "External rental $/Mtok (declining)", line: { color: COLORS.external, width: 3, dash: "dot" },
    },
  ];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    title: "Build-vs-buy: internal cost per Mtok vs declining external benchmark",
    xaxis: { title: "Year", dtick: 1 },
    yaxis: { title: "USD per million tokens", type: "log", tickformat: "$.2f" },
  }, { displaylogo: false, responsive: true });
}

export function renderLatencyFrontier(divId, points) {
  const traces = [{
    type: "scatter",
    mode: "lines+markers",
    x: points.map((p) => p.targetLatencyMs),
    y: points.map((p) => p.dollarsPerMtok),
    text: points.map((p) => `TP=${p.tp}, batch=${p.batch}, bound=${p.bound}`),
    line: { color: "#4F8EF7", width: 2 },
    marker: { size: 8, color: points.map((p) => p.tp), colorscale: "Viridis" },
    hovertemplate: "%{x} ms/tok target<br>$%{y:.2f}/Mtok<br>%{text}<extra></extra>",
    name: "Pareto frontier",
  }];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    title: "Latency–cost Pareto frontier (year 0)",
    xaxis: { title: "Latency target (ms per output token)", type: "log" },
    yaxis: { title: "USD per million output tokens", type: "log", tickformat: "$.2f" },
  }, { displaylogo: false, responsive: true });
}

export function renderTornado(divId, rows, baseTco) {
  // Sort by total span descending.
  rows.sort((a, b) => Math.abs(b.hi - b.lo) - Math.abs(a.hi - a.lo));
  const labels = rows.map((r) => prettyLeverName(r.lever));
  const lows = rows.map((r) => r.lo);
  const highs = rows.map((r) => r.hi);
  const traces = [
    { type: "bar", orientation: "h", x: lows, y: labels, name: "Low",
      marker: { color: "#4F8EF7" },
      hovertemplate: "Low value -> %{x:$.2s} delta TCO<extra></extra>" },
    { type: "bar", orientation: "h", x: highs, y: labels, name: "High",
      marker: { color: "#C8553D" },
      hovertemplate: "High value -> %{x:$.2s} delta TCO<extra></extra>" },
  ];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    title: `Tornado: TCO sensitivity (base = $${(baseTco / 1e6).toFixed(0)}M discounted)`,
    barmode: "overlay",
    xaxis: { title: "Δ Discounted TCO (USD)", tickformat: "$.2s", zeroline: true },
    yaxis: { automargin: true },
  }, { displaylogo: false, responsive: true });
}

export function renderDemandFan(divId, fan) {
  const traces = [
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.p90,
      name: "P90 demand", line: { color: "rgba(80,140,247,0.0)" }, showlegend: false,
    },
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.p10,
      name: "P10–P90 demand", fill: "tonexty", fillcolor: "rgba(80,140,247,0.25)",
      line: { color: "rgba(80,140,247,0.0)" },
    },
    {
      type: "scatter", mode: "lines+markers", x: fan.years, y: fan.p50,
      name: "Median demand", line: { color: "#4F8EF7", width: 3 },
    },
    {
      type: "scatter", mode: "lines+markers", x: fan.years, y: fan.capacity,
      name: "Provisioned capacity", line: { color: "#C8553D", width: 2, dash: "dash" },
    },
  ];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    title: "Demand uncertainty vs provisioned capacity",
    xaxis: { title: "Year", dtick: 1 },
    yaxis: { title: "GPU-years" },
  }, { displaylogo: false, responsive: true });
}

export function renderScaleUp(divId, plan) {
  const years = plan.yearly.map((y) => y.year);
  const minPod = plan.yearly.map((y) => y.minScaleUpGpus);
  const scaleUpDomain = plan.gpu.scaleup_domain || 8;
  const traces = [
    {
      type: "scatter", mode: "lines+markers", x: years, y: minPod,
      name: "Min coherent fabric size", line: { color: "#E08E45", width: 3 },
      hovertemplate: "Y%{x}: %{y} GPUs<extra></extra>",
    },
    {
      type: "scatter", mode: "lines", x: years, y: years.map(() => scaleUpDomain),
      name: `${plan.gpu.label} scale-up domain`, line: { color: "#7BB661", dash: "dash" },
    },
  ];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    title: "Coherent fabric size needed to host the year's frontier model",
    xaxis: { title: "Year", dtick: 1 },
    yaxis: { title: "GPUs in single scale-up domain", type: "log" },
  }, { displaylogo: false, responsive: true });
}

function prettyLeverName(k) {
  const map = {
    interactiveTokensPerDay: "Interactive demand",
    batchTokensPerDay: "Batch demand",
    demandGrowth: "Demand growth rate",
    modelGrowth: "Model size growth",
    algEfficiency: "Algorithmic efficiency gains",
    mfu: "Training MFU",
    utilization: "Inference utilization",
    pretrainFlopsPerYear: "Pretrain FLOPs/year",
    electricityPrice: "Electricity price",
    rentalShare: "Rental share",
  };
  return map[k] || k;
}

// ===========================================================================
// NEW: Workload-portfolio frame charts
// ===========================================================================

// Per-domain stacked bar of GPU-years over time. Replaces the legacy
// fleet-by-workload view as the primary "what's the fleet doing" chart,
// because for AI-for-science portfolios the legacy categories collapse.
export function renderDomainStack(divId, plan) {
  const years = plan.yearly.map((y) => `Y${y.year}`);
  const domainKeys = Object.keys(DOMAINS);
  const traces = domainKeys.map((dom) => ({
    type: "bar",
    x: years,
    y: plan.yearly.map((y) => y.gpuYearsByDomain?.[dom] || 0),
    name: DOMAINS[dom].label,
    marker: { color: DOMAINS[dom].color },
    hovertemplate: "%{y:,.0f} GPU-years<extra>%{fullData.name}</extra>",
  })).filter((t) => t.y.some((v) => v > 0));
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    barmode: "stack",
    title: "GPU-years by workload domain",
    xaxis: { title: "Year" },
    yaxis: { title: "GPU-years" },
  }, { displaylogo: false, responsive: true });
}

// Latent scaling-law uncertainty fan. Takes the output of scalingCaseFan().
// Shows low/median/high fleet trajectories so the buyer can see the band of
// outcomes that depend on whether unfitted scaling laws turn out compute-
// or data-bound.
export function renderScalingCaseFan(divId, fan) {
  const traces = [
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.high,
      name: "High (compute-bound)", line: { color: "rgba(200,85,61,0)" }, showlegend: false,
    },
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.low,
      name: "Low (data-bound) → High (compute-bound) band",
      fill: "tonexty", fillcolor: "rgba(200,85,61,0.20)",
      line: { color: "rgba(200,85,61,0)" },
    },
    {
      type: "scatter", mode: "lines+markers", x: fan.years, y: fan.median,
      name: "Median case (your selections)",
      line: { color: "#4F8EF7", width: 3 }, marker: { size: 8 },
    },
  ];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    title: "Latent scaling-law uncertainty: total GPUs across cases",
    xaxis: { title: "Year", dtick: 1 },
    yaxis: { title: "Total GPUs", type: "log" },
    annotations: [{
      xref: "paper", yref: "paper", x: 0.02, y: 0.98,
      text: "Spread reflects unfitted scaling laws<br>in non-LLM domains (data-bound vs compute-bound).",
      showarrow: false, font: { size: 10, color: "#9aa3b2" },
      align: "left", xanchor: "left", yanchor: "top",
    }],
  }, { displaylogo: false, responsive: true });
}

// FP64-required vs non-FP64 GPU-years over time. Surfaces how much of the
// fleet is doing classical-HPC-style (CFD, fusion, MD) work, which
// constrains SKU choice (excludes B200 due to FP64 downgrade).
export function renderFp64Split(divId, plan) {
  const years = plan.yearly.map((y) => `Y${y.year}`);
  const fp64Series = plan.yearly.map((y) => {
    let fp64 = 0, nonFp64 = 0;
    for (const ip of Object.values(y.itemPlans || {})) {
      const cls = WORKLOAD_CLASSES[ip.item?.classId];
      if (!cls || !ip.plan) continue;
      const gy = isFinite(ip.plan.gpuYears) ? ip.plan.gpuYears : 0;
      if (cls.fp64_required) fp64 += gy; else nonFp64 += gy;
    }
    return { fp64, nonFp64 };
  });
  const traces = [
    { type: "bar", x: years, y: fp64Series.map((p) => p.fp64),
      name: "FP64-required (HPC-class)", marker: { color: COLORS.fp64 } },
    { type: "bar", x: years, y: fp64Series.map((p) => p.nonFp64),
      name: "Non-FP64 (AI-class)", marker: { color: COLORS.non_fp64 } },
  ];
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    barmode: "stack",
    title: "FP64-bound vs general-AI fleet",
    xaxis: { title: "Year" },
    yaxis: { title: "GPU-years" },
  }, { displaylogo: false, responsive: true });
}

// Per-workload bottleneck heatmap. Each row is a workload class (only those
// active in the portfolio), each column is a year, and the cell shows where
// the constraint lives: memory bandwidth, compute, network, or FP64.
// Tells the buyer at a glance whether the next chip generation should target
// HBM, FP4/FP8 throughput, or NVLink/IB upgrades.
export function renderBoundHeatmap(divId, plan) {
  const activeIds = new Set();
  for (const y of plan.yearly) {
    for (const id of Object.keys(y.itemPlans || {})) {
      if ((y.itemPlans[id]?.plan?.gpuYears || 0) > 0) activeIds.add(id);
    }
  }
  const rowIds = [...activeIds];
  if (rowIds.length === 0) {
    Plotly.react(divId, [], { ...LAYOUT_BASE, title: "Per-workload bottleneck (no active workloads)" }, { displaylogo: false, responsive: true });
    return;
  }
  const years = plan.yearly.map((y) => `Y${y.year}`);

  // Numeric encoding for the bound type so plotly heatmap can color it.
  const boundIndex = { memory: 1, compute: 2, network: 3, fp64: 4, none: 0, OOM: 5 };
  const colorscale = [
    [0,    "#222"],
    [0.16, BOUND_COLORS.memory],
    [0.34, BOUND_COLORS.compute],
    [0.50, BOUND_COLORS.compute],
    [0.66, BOUND_COLORS.network],
    [0.82, BOUND_COLORS.fp64],
    [1.0,  "#7B2D26"],
  ];

  const z = rowIds.map((id) => plan.yearly.map((y) => {
    const ip = y.itemPlans?.[id];
    if (!ip || !ip.plan) return 0;
    const cls = WORKLOAD_CLASSES[id];
    const p = ip.plan;
    let bound = p.bound;
    if (!bound) {
      bound = p.decode?.bound || p.prefill?.bound || cls?.bound_default || "none";
    }
    if (cls?.fp64_required) bound = "fp64";    // override: FP64 dominates
    return boundIndex[bound] ?? 0;
  }));

  const text = rowIds.map((id) => plan.yearly.map((y) => {
    const ip = y.itemPlans?.[id];
    if (!ip || !ip.plan) return "—";
    const cls = WORKLOAD_CLASSES[id];
    const p = ip.plan;
    let bound = p.bound || p.decode?.bound || p.prefill?.bound || cls?.bound_default || "—";
    if (cls?.fp64_required) bound = "fp64";
    return bound;
  }));

  const yLabels = rowIds.map((id) => WORKLOAD_CLASSES[id]?.label || id);

  Plotly.react(divId, [{
    type: "heatmap",
    z, x: years, y: yLabels, text,
    texttemplate: "%{text}",
    textfont: { color: "#fff", size: 10 },
    colorscale, showscale: false,
    hovertemplate: "<b>%{y}</b><br>Year %{x}<br>Bound: %{text}<extra></extra>",
    zmin: 0, zmax: 5,
  }], {
    ...LAYOUT_BASE,
    title: "Per-workload bottleneck (where the constraint lives)",
    margin: { ...LAYOUT_BASE.margin, l: 220 },
    xaxis: { title: "" },
    yaxis: { title: "", automargin: true },
  }, { displaylogo: false, responsive: true });
}

// ===========================================================================
// Bonus: SKU mix over time (chip-mix frame).
// Stacked area showing how the fleet composition shifts year over year.
// ===========================================================================
export function renderSkuMix(divId, plan) {
  const years = plan.yearly.map((y) => `Y${y.year}`);
  const allSkus = new Set();
  for (const y of plan.yearly) {
    for (const sku of Object.keys(y.gpusBySku || {})) allSkus.add(sku);
  }
  const skuList = [...allSkus];
  if (skuList.length === 0) {
    Plotly.react(divId, [], { ...LAYOUT_BASE, title: "SKU mix (no workloads scheduled)" }, { displaylogo: false, responsive: true });
    return;
  }
  const traces = skuList.map((sku) => ({
    type: "bar",
    x: years,
    y: plan.yearly.map((y) => y.gpusBySku?.[sku] || 0),
    name: sku,
    marker: { color: SKU_COLORS[sku] || "#888" },
    hovertemplate: "%{y:,.0f} GPU-years<extra>%{fullData.name}</extra>",
  }));
  Plotly.react(divId, traces, {
    ...LAYOUT_BASE,
    barmode: "stack",
    title: "Fleet composition by GPU SKU",
    xaxis: { title: "Year" },
    yaxis: { title: "GPU-years" },
  }, { displaylogo: false, responsive: true });
}

export { fmt };
