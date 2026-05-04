// charts.js — Tufte-maximal, classified-dossier aesthetic
//
// Design principles applied:
//   1. Data-ink ratio maximized: no chart borders, no fills unless encoding data
//   2. Gridlines: horizontal only, hairline (#1E2740), no vertical
//   3. No Plotly titles — frame header is the title
//   4. Axis labels: minimal, uppercase, 10px, letter-spaced
//   5. Legend: inline annotations or right-side compact, not boxed
//   6. Colors: muted military palette — amber signal, slate blue structural,
//      desaturated series; high-contrast only for key signal
//   7. Markers: 5px, no border; lines 1.5px (thin = precise)
//   8. Hover: rich but not styled with rounded boxes
//   9. Bar gap: tight (0.25) for density
//  10. Annotations directly on data where possible

import { DOMAINS, WORKLOAD_CLASSES } from "./workload_classes.js";

// ── Palette ──────────────────────────────────────────────────────────────────
// Muted, desaturated — machine-spec sheet colors, not dashboard candy
const C = {
  amber:    "#D4860A",   // primary signal / highlight
  amber_lo: "#8C5800",   // dimmed amber
  slate:    "#4A6FA5",   // structural blue
  slate_lo: "#2A3F60",   // dimmed slate
  sage:     "#4A7C59",   // green-grey for "OK" signals
  rust:     "#8B3A2A",   // warning/stress
  stone:    "#5A6070",   // neutral series
  ghost:    "#2A2F3E",   // near-invisible fill
  fg:       "#C8D0DC",   // primary text on charts
  fg_dim:   "#5A6880",   // axis labels, secondary text
  bg:       "#080C14",   // plot bg
  grid:     "#131926",   // hairline grid
  zero:     "#1E2740",   // zero-line
};

// Domain colors — slightly desaturated from source to match palette
const DOMAIN_COLORS = {
  llm:           "#4A6FA5",
  biology:       "#4A7C59",
  physics:       "#8C6A1A",
  earth_obs:     "#2E6B7A",
  industry:      "#5A4A7A",
  defense_intel: "#7A3A2A",
};

const WORKLOAD_COLORS = {
  interactive_inference: "#4A6FA5",
  batch_inference:       "#4A7C59",
  rl:                    "#8C6A1A",
  training:              "#7A3A2A",
  finetune:              "#5A4A7A",
};

const SKU_COLORS = {
  A100_80GB:    "#3A5A30",
  H100_SXM:    "#4A6FA5",
  H200_SXM:    "#2A4F85",
  B200:         "#1A3060",
  GB200_NVL72:  "#0A2040",
  R100:         "#6A5A2A",
  MI300X:       "#7A4A1A",
  MI325X:       "#8A5A2A",
  MI355X:       "#9A6A3A",
  GAUDI3:       "#4A5A3A",
};

const TCO_COLORS = {
  capex_gpu:       "#4A6FA5",
  capex_network:   "#2E6B7A",
  capex_facility:  "#4A7C59",
  opex_electricity:"#8C6A1A",
  opex_facility:   "#5A4A7A",
  opex_rental:     "#5A6070",
};

const BOUND_COLORS = {
  memory:  "#4A6FA5",
  compute: "#4A7C59",
  network: "#8C6A1A",
  fp64:    "#7A3A2A",
  none:    "#1A1F2E",
  OOM:     "#5A1A1A",
};

// ── Layout base ───────────────────────────────────────────────────────────────
// Tufte: remove all chartjunk. No outer border on plot area. Hairline grid.
const L = {
  paper_bgcolor: C.bg,
  plot_bgcolor:  C.bg,
  font: {
    color:  C.fg_dim,
    family: "'DM Mono', 'Fira Mono', monospace",
    size:   10,
  },
  margin: { l: 52, r: 20, t: 18, b: 40 },
  legend: {
    bgcolor:     "rgba(0,0,0,0)",
    borderwidth: 0,
    font: { size: 9, color: C.fg_dim, family: "'DM Mono', monospace" },
    orientation: "h",
    x: 0, y: -0.18,
    xanchor: "left",
  },
  hoverlabel: {
    bgcolor:     "#0D1220",
    bordercolor: "#1E2740",
    font: { color: C.fg, size: 11, family: "'DM Mono', monospace" },
  },
  xaxis: axDefaults(),
  yaxis: axDefaults(),
};

function axDefaults(extra = {}) {
  return {
    showgrid:        false,
    zeroline:        false,
    linecolor:       C.zero,
    linewidth:       1,
    tickcolor:       C.zero,
    ticklen:         4,
    tickwidth:       1,
    tickfont:        { size: 9, color: C.fg_dim, family: "'DM Mono', monospace" },
    titlefont:       { size: 9, color: C.fg_dim, family: "'DM Mono', monospace" },
    title:           { standoff: 8 },
    ...extra,
  };
}

function yGridAxis(extra = {}) {
  return axDefaults({
    showgrid:    true,
    gridcolor:   C.grid,
    gridwidth:   1,
    ...extra,
  });
}

function layout(overrides = {}) {
  return { ...L, ...overrides };
}

// ── Config ────────────────────────────────────────────────────────────────────
const CFG = { displaylogo: false, responsive: true, displayModeBar: false };

// ── Helpers ───────────────────────────────────────────────────────────────────
export function fmt(n) {
  if (!isFinite(n)) return "—";
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(0);
}

function yearLabels(plan) { return plan.yearly.map((y) => `Y${y.year}`); }

// Thin horizontal-rule annotation label placed at the end of a line trace
function endLabel(x, y, text, color = C.fg_dim) {
  return {
    xref: "x", yref: "y",
    x, y,
    text: `<b>${text}</b>`,
    showarrow: false,
    font: { size: 9, color, family: "'DM Mono', monospace" },
    xanchor: "left",
    yanchor: "middle",
    xshift: 6,
  };
}

// Small "corner" text annotation (top-left of plot area)
function cornerNote(text) {
  return {
    xref: "paper", yref: "paper", x: 0.01, y: 0.99,
    text,
    showarrow: false,
    font: { size: 8, color: C.fg_dim, family: "'DM Mono', monospace" },
    align: "left", xanchor: "left", yanchor: "top",
  };
}

// ── Chart functions ───────────────────────────────────────────────────────────

export function renderFleetByWorkload(divId, plan) {
  const years = yearLabels(plan);
  const order = ["interactive_inference", "batch_inference", "rl", "training", "finetune"];
  const labels = { interactive_inference: "INTERACTIVE", batch_inference: "BATCH", rl: "RL", training: "PRETRAIN", finetune: "FINETUNE" };
  const traces = order
    .map((wl) => ({
      type: "bar",
      x: years,
      y: plan.yearly.map((y) => y.gpuYears[wl] || 0),
      name: labels[wl],
      marker: { color: WORKLOAD_COLORS[wl], line: { width: 0 } },
      hovertemplate: `<b>${labels[wl]}</b><br>%{y:,.0f} GPU-yr<extra></extra>`,
    }))
    .filter((t) => t.y.some((v) => v > 0));

  Plotly.react(divId, traces, layout({
    barmode: "stack",
    bargap: 0.3,
    xaxis: axDefaults(),
    yaxis: yGridAxis({ title: { text: "GPU-YEARS" } }),
    legend: { ...L.legend, traceorder: "normal" },
  }), CFG);
}

export function renderFacilities(divId, plan) {
  const years = yearLabels(plan);
  const tiers  = ["edge", "regional", "central"];
  const colors = { edge: C.amber, regional: C.slate, central: C.stone };
  const traces = tiers.map((tier) => ({
    type: "bar",
    x: years,
    y: plan.yearly.map((y) => y.tierFacilityCount[tier] || 0),
    name: tier.toUpperCase(),
    marker: { color: colors[tier], line: { width: 0 } },
    hovertemplate: `<b>${tier.toUpperCase()}</b><br>%{y} facilities<extra></extra>`,
  }));

  Plotly.react(divId, traces, layout({
    barmode: "stack",
    bargap: 0.3,
    xaxis: axDefaults(),
    yaxis: yGridAxis({ title: { text: "FACILITIES" }, dtick: 1 }),
  }), CFG);
}

export function renderPower(divId, plan) {
  const years = plan.yearly.map((y) => y.year);
  const vals  = plan.yearly.map((y) => +(y.totalKw / 1000).toFixed(2));
  const peakMw = Math.max(...vals);

  const traces = [{
    type: "scatter",
    mode: "lines+markers",
    x: years, y: vals,
    line:   { color: C.amber, width: 1.5, shape: "spline" },
    marker: { size: 5, color: C.amber, symbol: "circle" },
    fill:   "tozeroy",
    fillcolor: `${C.amber}18`,
    hovertemplate: "Y%{x} — %{y:.1f} MW<extra></extra>",
    showlegend: false,
  }];

  Plotly.react(divId, traces, layout({
    xaxis: axDefaults({ dtick: 1 }),
    yaxis: yGridAxis({ title: { text: "MW (INCL. PUE)" } }),
    annotations: [
      cornerNote(`PEAK ${peakMw.toFixed(1)} MW`),
      endLabel(years[years.length - 1], vals[vals.length - 1], `${vals[vals.length-1].toFixed(1)} MW`, C.amber),
    ],
  }), CFG);
}

export function renderTcoStack(divId, plan) {
  const years = yearLabels(plan);
  const series = [
    { key: "capex_gpu",       getter: (y) => y.capex.gpu,        name: "GPU CAPEX"     },
    { key: "capex_network",   getter: (y) => y.capex.network,    name: "NETWORK CAPEX" },
    { key: "capex_facility",  getter: (y) => y.capex.facility,   name: "FACILITY CAPEX"},
    { key: "opex_electricity",getter: (y) => y.opex.electricity, name: "POWER OPEX"   },
    { key: "opex_facility",   getter: (y) => y.opex.facility,    name: "FAC. OPEX"    },
    { key: "opex_rental",     getter: (y) => y.opex.rental,      name: "RENTAL OPEX"  },
  ];

  const traces = series
    .map((s) => ({
      type: "bar",
      x: years,
      y: plan.yearly.map((y) => s.getter(y) || 0),
      name: s.name,
      marker: { color: TCO_COLORS[s.key], line: { width: 0 } },
      hovertemplate: `<b>${s.name}</b><br>$%{y:,.0f}<extra></extra>`,
    }))
    .filter((t) => t.y.some((v) => v > 0));

  const totals = plan.yearly.map((y) =>
    (y.capex.gpu||0)+(y.capex.network||0)+(y.capex.facility||0)+(y.opex.electricity||0)+(y.opex.facility||0)+(y.opex.rental||0)
  );
  const peakTco = Math.max(...totals);

  Plotly.react(divId, traces, layout({
    barmode: "stack",
    bargap:  0.3,
    xaxis:   axDefaults(),
    yaxis:   yGridAxis({ title: { text: "USD / YEAR" }, tickformat: "$.2s" }),
    annotations: [ cornerNote(`PEAK YEAR $${(peakTco/1e6).toFixed(0)}M`) ],
    legend:  { ...L.legend },
  }), CFG);
}

export function renderBuildVsBuy(divId, plan) {
  const years = plan.yearly.map((y) => y.year);
  const intVals = plan.yearly.map((y) => y.internalDollarsPerMtok);
  const extVals = plan.yearly.map((y) => y.externalDollarsPerMtok);
  const crossYear = years.find((yr, i) => intVals[i] < extVals[i]);

  const traces = [
    {
      type: "scatter", mode: "lines+markers", x: years, y: intVals,
      name: "INTERNAL",
      line:   { color: C.slate, width: 1.5 },
      marker: { size: 5, color: C.slate },
      hovertemplate: "Y%{x} internal: $%{y:.2f}/Mtok<extra></extra>",
    },
    {
      type: "scatter", mode: "lines+markers", x: years, y: extVals,
      name: "EXTERNAL (RENTAL)",
      line:   { color: C.amber, width: 1.5, dash: "dot" },
      marker: { size: 5, color: C.amber, symbol: "diamond" },
      hovertemplate: "Y%{x} external: $%{y:.2f}/Mtok<extra></extra>",
    },
  ];

  const annotations = [
    endLabel(years[years.length-1], intVals[intVals.length-1], "INTERNAL", C.slate),
    endLabel(years[years.length-1], extVals[extVals.length-1], "EXTERNAL", C.amber),
  ];
  if (crossYear !== undefined) {
    annotations.push(cornerNote(`CROSSOVER: Y${crossYear}`));
  }

  Plotly.react(divId, traces, layout({
    xaxis: axDefaults({ dtick: 1 }),
    yaxis: yGridAxis({ title: { text: "$/MTOK (LOG)" }, type: "log", tickformat: "$.2f" }),
    showlegend: false,
    annotations,
  }), CFG);
}

export function renderLatencyFrontier(divId, points) {
  if (!points || points.length === 0) return;

  // Color by "bound" type for diagnostic read
  const boundColorMap = { memory: C.slate, compute: C.sage, network: C.amber, fp64: C.rust };
  const markerColors = points.map((p) => boundColorMap[p.bound] || C.stone);

  const traces = [{
    type: "scatter",
    mode: "lines+markers",
    x: points.map((p) => p.targetLatencyMs),
    y: points.map((p) => p.dollarsPerMtok),
    text: points.map((p) => `TP${p.tp} · B${p.batch} · ${(p.bound||"?").toUpperCase()}`),
    line:   { color: C.stone, width: 1, shape: "spline" },
    marker: { size: 6, color: markerColors, symbol: "circle", line: { width: 0 } },
    hovertemplate: "%{x}ms → $%{y:.3f}/Mtok<br>%{text}<extra></extra>",
    showlegend: false,
  }];

  // Annotate the fastest and cheapest points
  const fastest = points[0];
  const cheapest = points[points.length - 1];

  Plotly.react(divId, traces, layout({
    xaxis: axDefaults({ title: { text: "LATENCY TARGET (MS/TOK, LOG)" }, type: "log" }),
    yaxis: yGridAxis({ title: { text: "$/MTOK (LOG)" }, type: "log", tickformat: "$.2f" }),
    annotations: [
      {
        xref: "x", yref: "y", x: fastest.targetLatencyMs, y: fastest.dollarsPerMtok,
        text: `${fastest.targetLatencyMs}ms<br>$${fastest.dollarsPerMtok.toFixed(2)}`,
        showarrow: true, arrowhead: 0, arrowcolor: C.fg_dim, arrowwidth: 1,
        ax: 30, ay: -30,
        font: { size: 8, color: C.amber, family: "'DM Mono', monospace" },
      },
      {
        xref: "x", yref: "y", x: cheapest.targetLatencyMs, y: cheapest.dollarsPerMtok,
        text: `${cheapest.targetLatencyMs}ms<br>$${cheapest.dollarsPerMtok.toFixed(2)}`,
        showarrow: true, arrowhead: 0, arrowcolor: C.fg_dim, arrowwidth: 1,
        ax: -30, ay: 30,
        font: { size: 8, color: C.sage, family: "'DM Mono', monospace" },
      },
      cornerNote("DOT COLOR = BOUND TYPE  ■MEM ■COMP ■NET ■FP64"),
    ],
  }), CFG);
}

export function renderTornado(divId, rows, baseTco) {
  rows = [...rows].sort((a, b) => Math.abs(b.hi - b.lo) - Math.abs(a.hi - a.lo));
  const labels = rows.map((r) => prettyLeverName(r.lever));

  // Diverging bars from zero baseline
  const traces = [
    {
      type: "bar", orientation: "h",
      x: rows.map((r) => r.lo < 0 ? r.lo : 0),
      y: labels,
      name: "LOW SCENARIO",
      marker: { color: C.slate, line: { width: 0 } },
      hovertemplate: "LOW: %{x:+$.2s} Δ TCO<extra></extra>",
    },
    {
      type: "bar", orientation: "h",
      x: rows.map((r) => r.hi > 0 ? r.hi : 0),
      y: labels,
      name: "HIGH SCENARIO",
      marker: { color: C.amber, line: { width: 0 } },
      hovertemplate: "HIGH: %{x:+$.2s} Δ TCO<extra></extra>",
    },
  ];

  Plotly.react(divId, traces, layout({
    barmode:  "overlay",
    bargap:   0.35,
    xaxis: axDefaults({
      title:      { text: "Δ DISCOUNTED TCO (USD)" },
      tickformat: "$.2s",
      zeroline:   true,
      zerolinecolor: C.fg_dim,
      zerolinewidth: 1,
    }),
    yaxis: axDefaults({ automargin: true }),
    annotations: [ cornerNote(`BASE $${(baseTco/1e6).toFixed(0)}M`) ],
  }), CFG);
}

export function renderDemandFan(divId, fan) {
  const traces = [
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.p90,
      line: { color: "rgba(0,0,0,0)" }, showlegend: false,
      hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.p10,
      name: "P10–P90 BAND",
      fill: "tonexty", fillcolor: `${C.slate}28`,
      line: { color: "rgba(0,0,0,0)" },
      hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "lines+markers", x: fan.years, y: fan.p50,
      name: "MEDIAN",
      line:   { color: C.slate, width: 1.5 },
      marker: { size: 4, color: C.slate },
      hovertemplate: "Y%{x} median: %{y:,.0f} GPU-yr<extra></extra>",
    },
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.capacity,
      name: "PROVISIONED",
      line: { color: C.amber, width: 1.5, dash: "dash" },
      hovertemplate: "Y%{x} provisioned: %{y:,.0f}<extra></extra>",
    },
  ];

  const lastYear = fan.years[fan.years.length - 1];
  Plotly.react(divId, traces, layout({
    xaxis: axDefaults({ dtick: 1 }),
    yaxis: yGridAxis({ title: { text: "GPU-YEARS" } }),
    annotations: [
      endLabel(lastYear, fan.p50[fan.p50.length-1],   "MEDIAN",      C.slate),
      endLabel(lastYear, fan.capacity[fan.capacity.length-1], "PROVISIONED", C.amber),
    ],
    showlegend: false,
  }), CFG);
}

export function renderScaleUp(divId, plan) {
  const years = plan.yearly.map((y) => y.year);
  const minPod = plan.yearly.map((y) => y.minScaleUpGpus);
  const domain = plan.gpu?.scaleup_domain || 8;

  const traces = [
    {
      type: "scatter", mode: "lines+markers", x: years, y: minPod,
      name: "REQ. FABRIC",
      line:   { color: C.amber, width: 1.5 },
      marker: { size: 5, color: C.amber },
      hovertemplate: "Y%{x}: %{y} GPUs required<extra></extra>",
    },
    {
      type: "scatter", mode: "lines", x: years, y: years.map(() => domain),
      name: "SKU DOMAIN",
      line: { color: C.sage, width: 1, dash: "dash" },
      hovertemplate: `SKU domain: ${domain} GPUs<extra></extra>`,
    },
  ];

  Plotly.react(divId, traces, layout({
    xaxis: axDefaults({ dtick: 1 }),
    yaxis: yGridAxis({ title: { text: "GPUS (LOG)" }, type: "log" }),
    annotations: [
      endLabel(years[years.length-1], minPod[minPod.length-1], "REQUIRED", C.amber),
      endLabel(years[years.length-1], domain, "SKU MAX", C.sage),
    ],
    showlegend: false,
  }), CFG);
}

export function renderDomainStack(divId, plan) {
  const years = yearLabels(plan);
  const domainKeys = Object.keys(DOMAINS);
  const traces = domainKeys
    .map((dom) => ({
      type: "bar",
      x: years,
      y: plan.yearly.map((y) => y.gpuYearsByDomain?.[dom] || 0),
      name: DOMAINS[dom].label.toUpperCase().split(" ")[0],
      marker: { color: DOMAIN_COLORS[dom] || C.stone, line: { width: 0 } },
      hovertemplate: `<b>${DOMAINS[dom].label}</b><br>%{y:,.0f} GPU-yr<extra></extra>`,
    }))
    .filter((t) => t.y.some((v) => v > 0));

  Plotly.react(divId, traces, layout({
    barmode: "stack",
    bargap:  0.3,
    xaxis:   axDefaults(),
    yaxis:   yGridAxis({ title: { text: "GPU-YEARS" } }),
  }), CFG);
}

export function renderScalingCaseFan(divId, fan) {
  const traces = [
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.high,
      line: { color: "rgba(0,0,0,0)" }, showlegend: false, hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "lines", x: fan.years, y: fan.low,
      name: "UNCERTAINTY BAND",
      fill: "tonexty", fillcolor: `${C.rust}22`,
      line: { color: "rgba(0,0,0,0)" }, hoverinfo: "skip",
    },
    {
      type: "scatter", mode: "lines+markers", x: fan.years, y: fan.median,
      name: "MEDIAN CASE",
      line:   { color: C.slate, width: 1.5 },
      marker: { size: 4, color: C.slate },
      hovertemplate: "Y%{x} median: %{y:,.0f} GPUs<extra></extra>",
    },
  ];

  const lastYr = fan.years[fan.years.length-1];
  Plotly.react(divId, traces, layout({
    xaxis: axDefaults({ dtick: 1 }),
    yaxis: yGridAxis({ title: { text: "TOTAL GPUs (LOG)" }, type: "log" }),
    annotations: [
      endLabel(lastYr, fan.high[fan.high.length-1], "HIGH", C.rust),
      endLabel(lastYr, fan.median[fan.median.length-1], "MEDIAN", C.slate),
      endLabel(lastYr, fan.low[fan.low.length-1], "LOW", C.fg_dim),
      cornerNote("BAND = UNFITTED SCALING LAWS (DATA-BOUND → COMPUTE-BOUND)"),
    ],
    showlegend: false,
  }), CFG);
}

export function renderFp64Split(divId, plan) {
  const years = yearLabels(plan);
  const series = plan.yearly.map((y) => {
    let fp64 = 0, ai = 0;
    for (const ip of Object.values(y.itemPlans || {})) {
      const cls = WORKLOAD_CLASSES[ip.item?.classId];
      if (!cls || !ip.plan) continue;
      const gy = isFinite(ip.plan.gpuYears) ? ip.plan.gpuYears : 0;
      if (cls.fp64_required) fp64 += gy; else ai += gy;
    }
    return { fp64, ai };
  });

  const traces = [
    {
      type: "bar", x: years,
      y: series.map((s) => s.fp64),
      name: "FP64 / HPC",
      marker: { color: C.rust, line: { width: 0 } },
      hovertemplate: "FP64: %{y:,.0f} GPU-yr<extra></extra>",
    },
    {
      type: "bar", x: years,
      y: series.map((s) => s.ai),
      name: "AI / ML",
      marker: { color: C.slate, line: { width: 0 } },
      hovertemplate: "AI/ML: %{y:,.0f} GPU-yr<extra></extra>",
    },
  ];

  Plotly.react(divId, traces, layout({
    barmode: "stack",
    bargap:  0.3,
    xaxis:   axDefaults(),
    yaxis:   yGridAxis({ title: { text: "GPU-YEARS" } }),
    annotations: [
      cornerNote("RED = FP64-BOUND (B200 EXCLUDED)"),
    ],
  }), CFG);
}

export function renderBoundHeatmap(divId, plan) {
  const activeIds = new Set();
  for (const y of plan.yearly) {
    for (const id of Object.keys(y.itemPlans || {})) {
      if ((y.itemPlans[id]?.plan?.gpuYears || 0) > 0) activeIds.add(id);
    }
  }
  const rowIds = [...activeIds];
  if (rowIds.length === 0) {
    Plotly.react(divId, [], layout({ title: undefined }), CFG);
    return;
  }
  const years = yearLabels(plan);
  const boundIndex = { none: 0, memory: 1, compute: 2, network: 3, fp64: 4, OOM: 5 };

  // Use a high-contrast stepped colorscale matching bound types
  const colorscale = [
    [0,     "#0D1220"],         // none
    [0.167, BOUND_COLORS.memory],
    [0.334, BOUND_COLORS.memory],
    [0.334, BOUND_COLORS.compute],
    [0.501, BOUND_COLORS.compute],
    [0.501, BOUND_COLORS.network],
    [0.668, BOUND_COLORS.network],
    [0.668, BOUND_COLORS.fp64],
    [0.835, BOUND_COLORS.fp64],
    [0.835, BOUND_COLORS.OOM],
    [1.0,   BOUND_COLORS.OOM],
  ];

  const z = rowIds.map((id) =>
    plan.yearly.map((y) => {
      const ip = y.itemPlans?.[id];
      if (!ip?.plan) return 0;
      const cls = WORKLOAD_CLASSES[id];
      let bound = ip.plan.bound || ip.plan.decode?.bound || ip.plan.prefill?.bound || cls?.bound_default || "none";
      if (cls?.fp64_required) bound = "fp64";
      return boundIndex[bound] ?? 0;
    })
  );

  const textZ = rowIds.map((id) =>
    plan.yearly.map((y) => {
      const ip = y.itemPlans?.[id];
      if (!ip?.plan) return "";
      const cls = WORKLOAD_CLASSES[id];
      let b = ip.plan.bound || ip.plan.decode?.bound || ip.plan.prefill?.bound || cls?.bound_default || "—";
      if (cls?.fp64_required) b = "fp64";
      return b.toUpperCase().slice(0,3);
    })
  );

  const yLabels = rowIds.map((id) => {
    const label = WORKLOAD_CLASSES[id]?.label || id;
    // Truncate long labels for the heatmap
    return label.length > 32 ? label.slice(0, 30) + "…" : label;
  });

  Plotly.react(divId, [{
    type: "heatmap",
    z, x: years, y: yLabels,
    text: textZ,
    texttemplate: "%{text}",
    textfont: { color: "rgba(255,255,255,0.7)", size: 8, family: "'DM Mono', monospace" },
    colorscale, showscale: false,
    hovertemplate: "<b>%{y}</b> · %{x}<br>BOUND: %{text}<extra></extra>",
    zmin: 0, zmax: 5,
    xgap: 1, ygap: 1,
  }], layout({
    margin: { l: 200, r: 20, t: 18, b: 40 },
    xaxis: axDefaults(),
    yaxis: axDefaults({ automargin: true, tickfont: { size: 8, color: C.fg_dim, family: "'DM Mono', monospace" } }),
    annotations: [
      cornerNote("MEM  ■  COMP  ■  NET  ■  FP64"),
    ],
  }), CFG);
}

export function renderSkuMix(divId, plan) {
  const years = yearLabels(plan);
  const allSkus = new Set();
  for (const y of plan.yearly) {
    for (const sku of Object.keys(y.gpusBySku || {})) allSkus.add(sku);
  }
  const skuList = [...allSkus];
  if (skuList.length === 0) {
    Plotly.react(divId, [], layout({}), CFG);
    return;
  }

  const palette = [C.slate, C.amber, C.sage, C.rust, C.stone, "#3A5A6A", "#6A3A5A", "#5A6A3A"];
  const traces = skuList.map((sku, i) => ({
    type: "bar",
    x: years,
    y: plan.yearly.map((y) => y.gpusBySku?.[sku] || 0),
    name: sku.replace(/_/g, " "),
    marker: { color: SKU_COLORS[sku] || palette[i % palette.length], line: { width: 0 } },
    hovertemplate: `<b>${sku}</b><br>%{y:,.0f} GPU-yr<extra></extra>`,
  }));

  Plotly.react(divId, traces, layout({
    barmode: "stack",
    bargap:  0.3,
    xaxis:   axDefaults(),
    yaxis:   yGridAxis({ title: { text: "GPU-YEARS" } }),
  }), CFG);
}

// ── Lever name map ────────────────────────────────────────────────────────────
function prettyLeverName(k) {
  const map = {
    interactiveTokensPerDay: "INTERACTIVE DEMAND",
    batchTokensPerDay:       "BATCH DEMAND",
    demandGrowth:            "DEMAND GROWTH",
    modelGrowth:             "MODEL SIZE GROWTH",
    algEfficiency:           "ALG. EFFICIENCY",
    mfu:                     "TRAINING MFU",
    utilization:             "INFERENCE UTIL.",
    pretrainFlopsPerYear:    "PRETRAIN FLOPs",
    electricityPrice:        "ELECTRICITY PRICE",
    rentalShare:             "RENTAL SHARE",
  };
  return map[k] || k.toUpperCase();
}
