// Plotly chart helpers. Each function takes a plan object and a target div id.

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

export { fmt };
