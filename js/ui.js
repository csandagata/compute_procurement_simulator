// Wires DOM controls to the model and charts.

import { GPUS } from "./gpus.js";
import { SCENARIOS, DEFAULT_SCENARIO } from "./scenarios.js";
import {
  buildPlan,
  tornadoBattery,
  demandFan,
  latencyCostFrontier,
  scalingCaseFan,
} from "./model.js";
import {
  renderFleetByWorkload,
  renderFacilities,
  renderPower,
  renderTcoStack,
  renderBuildVsBuy,
  renderLatencyFrontier,
  renderTornado,
  renderDemandFan,
  renderScaleUp,
  renderDomainStack,
  renderScalingCaseFan,
  renderFp64Split,
  renderBoundHeatmap,
  renderSkuMix,
  fmt,
} from "./charts.js";
import { KV_SCHEMES, KV_SCHEME_LABELS } from "./kv.js";
import { WORKLOAD_CLASSES, DOMAINS, workloadsByDomain, SCALING_CASES } from "./workload_classes.js";
import { makePortfolioItem } from "./portfolio.js";
import { HATS, DEFAULT_HAT, FRAME_LABELS, KPI_META } from "./hats.js";

// Spec for every scalar input. The portfolio is rendered separately.
const CONTROLS = [
  { group: "Planning horizon", key: "horizonYears", label: "Years", type: "number", min: 1, max: 10, step: 1 },
  { group: "Planning horizon", key: "discountRate", label: "Discount rate", type: "number", min: 0, max: 0.25, step: 0.01 },
  { group: "Planning horizon", key: "refreshYears", label: "Refresh cycle (yrs)", type: "number", min: 2, max: 7, step: 1 },

  { group: "Hardware", key: "gpuKey", label: "Headline GPU SKU (legacy)", type: "select", options: Object.keys(GPUS) },
  { group: "Hardware", key: "allowedGpus", label: "Allowed GPU SKUs", type: "multicheckbox", options: Object.keys(GPUS) },
  { group: "Hardware", key: "quantization", label: "Quantization (LLM)", type: "select", options: ["fp16", "fp8", "int4"] },
  { group: "Hardware", key: "networkTier", label: "Network fabric tier", type: "select", options: ["scale_up_only", "small_pod", "large_pod", "full_fabric"] },
  { group: "Hardware", key: "pue", label: "Datacenter PUE", type: "number", min: 1.05, max: 2.0, step: 0.01 },
  { group: "Hardware", key: "electricityPrice", label: "Electricity ($/kWh)", type: "number", min: 0.02, max: 0.30, step: 0.005 },

  { group: "Model (LLM)", key: "paramsB", label: "LLM size (B params)", type: "number", min: 1, max: 5000, step: 1 },
  { group: "Model (LLM)", key: "activeFrac", label: "Active param fraction (MoE)", type: "number", min: 0.05, max: 1.0, step: 0.05 },
  { group: "Model (LLM)", key: "contextLen", label: "Avg input length (tokens)", type: "number", min: 1000, max: 200_000, step: 1000 },
  { group: "Model (LLM)", key: "outputLen", label: "Avg output length (tokens)", type: "number", min: 50, max: 10_000, step: 50 },

  { group: "Architecture & runtime", key: "kvScheme", label: "KV-cache scheme", type: "select", options: KV_SCHEMES },
  { group: "Architecture & runtime", key: "mlaCompressionDim", label: "MLA compression dim", type: "number", min: 64, max: 2048, step: 32 },
  { group: "Architecture & runtime", key: "kvSchemeOverride", label: "Use explicit KV bytes/token", type: "checkbox" },
  { group: "Architecture & runtime", key: "kvBytesPerToken", label: "↳ explicit KV bytes/token", type: "number", min: 10_000, max: 1_000_000, step: 10_000 },
  { group: "Architecture & runtime", key: "enableSpecDec", label: "Enable speculative decoding", type: "checkbox" },
  { group: "Architecture & runtime", key: "specDecAcceptanceProb", label: "Spec-dec acceptance prob", type: "number", min: 0, max: 0.99, step: 0.05 },
  { group: "Architecture & runtime", key: "specDecGammaMax", label: "Spec-dec γ (draft tokens)", type: "number", min: 1, max: 10, step: 1 },
  { group: "Architecture & runtime", key: "enableQueueing", label: "Apply queueing-derived util cap", type: "checkbox" },

  { group: "Operations", key: "mfu", label: "Training MFU", type: "number", min: 0.05, max: 0.7, step: 0.01 },
  { group: "Operations", key: "utilization", label: "Inference fleet utilization (cap)", type: "number", min: 0.2, max: 0.95, step: 0.01 },
  { group: "Operations", key: "headroomFactor", label: "Capacity headroom factor", type: "number", min: 1.0, max: 2.5, step: 0.05 },
  { group: "Operations", key: "demandGrowth", label: "Default YoY demand growth", type: "number", min: 0, max: 3.0, step: 0.05 },
  { group: "Operations", key: "modelGrowth", label: "LLM model size YoY growth", type: "number", min: 0, max: 2.0, step: 0.05 },
  { group: "Operations", key: "algEfficiency", label: "Alg efficiency YoY gain", type: "number", min: 0, max: 2.0, step: 0.05 },

  { group: "Strategy", key: "rentalShare", label: "Rental share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "edgeFacilityShare", label: "Edge facility GPU share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "regionalFacilityShare", label: "Regional facility GPU share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "centralFacilityShare", label: "Central facility GPU share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "facilityMaxMw", label: "Max MW per facility", type: "number", min: 5, max: 500, step: 5 },

  { group: "External benchmark", key: "externalInitialPrice", label: "External $/Mtok today", type: "number", min: 0.1, max: 100, step: 0.1 },
  { group: "External benchmark", key: "externalDeclineMult", label: "External price decline (x/yr)", type: "number", min: 1, max: 50, step: 0.5 },
];

let currentInputs = {};
let currentHat = DEFAULT_HAT;
// currentInputs.portfolio is an array of portfolio items; mirrored in DOM by the portfolio editor.

// =====================================================================
// KPI / hat-pill rendering
// =====================================================================

function buildKpiRow() {
  const row = document.getElementById("kpi-row");
  row.innerHTML = "";
  const order = ["kpi-peak-gpus", "kpi-peak-mw", "kpi-peak-facilities", "kpi-tco", "kpi-internal-price", "kpi-fabric"];
  for (const id of order) {
    const meta = KPI_META[id];
    if (!meta) continue;
    const tile = document.createElement("div");
    tile.className = "kpi";
    tile.dataset.kpiId = id;
    tile.innerHTML = `
      <div class="kpi-header">
        <div class="kpi-label">${meta.label}</div>
        <span class="info-icon" title="${meta.explainer.replace(/"/g, '&quot;')}">i</span>
      </div>
      <div class="kpi-value" id="${id}">—</div>
      <div class="kpi-hats">${meta.hats.map((h) => `<span class="hat-tag" data-hat="${h}">${HATS[h].short}</span>`).join("")}</div>
    `;
    row.appendChild(tile);
  }
}

function buildHatPills() {
  const wrap = document.getElementById("hat-pills");
  wrap.innerHTML = "";
  for (const [id, hat] of Object.entries(HATS)) {
    const pill = document.createElement("button");
    pill.className = "hat-pill";
    pill.dataset.hat = id;
    pill.innerHTML = `<span class="hat-icon">${hat.icon}</span> ${hat.short}`;
    pill.title = hat.label;
    pill.addEventListener("click", () => setHat(id));
    wrap.appendChild(pill);
  }
}

function setHat(hatId) {
  currentHat = hatId;
  // Pill active state
  for (const el of document.querySelectorAll(".hat-pill")) {
    el.classList.toggle("active", el.dataset.hat === hatId);
  }
  // Frame highlighting
  const hat = HATS[hatId];
  for (const el of document.querySelectorAll(".frame")) {
    const isPrimary = hat.primaryFrames.includes(el.dataset.frame);
    el.classList.toggle("primary-for-hat", isPrimary);
    el.classList.toggle("secondary-for-hat", hatId !== "all" && !isPrimary);
  }
  // KPI emphasis
  for (const el of document.querySelectorAll(".kpi")) {
    const isKey = hat.keyKpis.includes(el.dataset.kpiId);
    el.classList.toggle("primary-for-hat", hatId !== "all" && isKey);
    el.classList.toggle("secondary-for-hat", hatId !== "all" && !isKey);
  }
  // Hat tag highlight inside KPIs
  for (const tag of document.querySelectorAll(".hat-tag")) {
    tag.classList.toggle("active", tag.dataset.hat === hatId);
  }
  // Hat blurb
  document.getElementById("hat-blurb").textContent = hat.blurb;
  // Hat questions panel
  const qs = document.getElementById("hat-questions");
  if (hatId === "all" || !hat.questions) {
    qs.innerHTML = "";
    qs.classList.remove("visible");
  } else {
    qs.classList.add("visible");
    qs.innerHTML = `
      <div class="hq-title"><span class="hq-icon">${hat.icon}</span> Through the <strong>${hat.label}</strong> lens, ask:</div>
      <ul class="hq-list">${hat.questions.map((q) => `<li>${q}</li>`).join("")}</ul>
    `;
  }
}

function buildControlsDom() {
  const container = document.getElementById("controls");
  container.innerHTML = "";

  // Portfolio editor first (expanded by default).
  container.appendChild(buildPortfolioEditor());

  // Then scalar control groups.
  const groups = {};
  for (const c of CONTROLS) {
    if (!groups[c.group]) groups[c.group] = [];
    groups[c.group].push(c);
  }
  for (const [groupName, controls] of Object.entries(groups)) {
    const section = document.createElement("section");
    section.className = "control-group";
    section.innerHTML = `<h3>${groupName}</h3>`;
    for (const c of controls) {
      section.appendChild(buildControl(c));
    }
    container.appendChild(section);
  }
}

// Portfolio editor: collapsible per-domain panels, one row per workload class.
function buildPortfolioEditor() {
  const section = document.createElement("section");
  section.className = "control-group portfolio-editor";
  section.innerHTML = `
    <h3>Workload portfolio</h3>
    <div class="hint">
      Compose a mix of workloads. Each row's <em>scaling case</em> selects how
      compute demand grows over time when the underlying scaling law is unfitted —
      <code>low</code> is data-bound / saturating, <code>high</code> is compute-bound.
    </div>`;
  const byDomain = workloadsByDomain();
  for (const [domain, items] of Object.entries(byDomain)) {
    const dom = DOMAINS[domain];
    const panel = document.createElement("details");
    panel.open = true;
    panel.className = "domain-panel";
    panel.style.borderLeft = `3px solid ${dom.color}`;
    const summary = document.createElement("summary");
    summary.innerHTML = `<span class="dom-dot" style="background:${dom.color}"></span> ${dom.label}`;
    panel.appendChild(summary);
    for (const cls of items) {
      panel.appendChild(buildPortfolioRow(cls));
    }
    section.appendChild(panel);
  }
  return section;
}

function buildPortfolioRow(cls) {
  const row = document.createElement("div");
  row.className = "portfolio-row";
  row.dataset.classId = cls.id;

  const labelWrap = document.createElement("label");
  labelWrap.className = "pf-toplabel";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.className = "pf-enable";
  cb.dataset.classId = cls.id;
  cb.addEventListener("change", onPortfolioChange);
  labelWrap.appendChild(cb);
  const labelSpan = document.createElement("span");
  labelSpan.textContent = cls.label;
  labelSpan.title = cls.notes || "";
  labelWrap.appendChild(labelSpan);
  row.appendChild(labelWrap);

  const inputs = document.createElement("div");
  inputs.className = "pf-inputs";

  const demandInput = document.createElement("input");
  demandInput.type = "number";
  demandInput.className = "pf-demand";
  demandInput.dataset.classId = cls.id;
  demandInput.min = 0;
  demandInput.step = "any";
  demandInput.placeholder = cls.default_demand?.label || "units/year";
  demandInput.title = cls.default_demand?.label || "units/year";
  demandInput.addEventListener("input", debounce(onPortfolioChange, 250));
  inputs.appendChild(demandInput);

  const scalingSelect = document.createElement("select");
  scalingSelect.className = "pf-scaling";
  scalingSelect.dataset.classId = cls.id;
  // Only include "fitted" for LLM domain; others have unfitted scaling laws.
  const cases = cls.domain === "llm" ? SCALING_CASES : SCALING_CASES.filter((s) => s !== "fitted");
  for (const s of cases) {
    const o = document.createElement("option");
    o.value = s; o.textContent = s;
    scalingSelect.appendChild(o);
  }
  scalingSelect.addEventListener("change", onPortfolioChange);
  inputs.appendChild(scalingSelect);

  row.appendChild(inputs);

  const meta = document.createElement("div");
  meta.className = "pf-meta";
  const bound = cls.bound_default || cls.compute_pattern;
  const fp64 = cls.fp64_required ? " · FP64" : "";
  const ecosystem = cls.ecosystem !== "cuda_or_rocm" ? ` · ${cls.ecosystem}` : "";
  meta.textContent = `${cls.compute_pattern} · ${bound}${fp64}${ecosystem}`;
  row.appendChild(meta);

  return row;
}

function applyPortfolioToDom(portfolio) {
  // First clear all rows
  for (const cls of Object.values(WORKLOAD_CLASSES)) {
    const en = document.querySelector(`.pf-enable[data-class-id="${cls.id}"]`);
    const dm = document.querySelector(`.pf-demand[data-class-id="${cls.id}"]`);
    const sc = document.querySelector(`.pf-scaling[data-class-id="${cls.id}"]`);
    if (!en) continue;
    en.checked = false;
    dm.value = "";
    sc.value = "median";
  }
  if (!Array.isArray(portfolio)) return;
  for (const item of portfolio) {
    const en = document.querySelector(`.pf-enable[data-class-id="${item.classId}"]`);
    const dm = document.querySelector(`.pf-demand[data-class-id="${item.classId}"]`);
    const sc = document.querySelector(`.pf-scaling[data-class-id="${item.classId}"]`);
    if (!en) continue;
    en.checked = item.enabled !== false;
    dm.value = item.unitsPerYear;
    sc.value = item.scalingCase || "median";
  }
}

function readPortfolioFromDom() {
  const items = [];
  for (const cls of Object.values(WORKLOAD_CLASSES)) {
    const en = document.querySelector(`.pf-enable[data-class-id="${cls.id}"]`);
    const dm = document.querySelector(`.pf-demand[data-class-id="${cls.id}"]`);
    const sc = document.querySelector(`.pf-scaling[data-class-id="${cls.id}"]`);
    if (!en || !en.checked) continue;
    const val = parseFloat(dm.value);
    if (!isFinite(val) || val <= 0) continue;
    items.push(makePortfolioItem(cls.id, {
      unitsPerYear: val,
      scalingCase: sc.value,
    }));
  }
  return items;
}

function makeCheckOption(key, opt) {
  const wrap = document.createElement("label");
  wrap.className = "ctl-check-mini";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.dataset.key = key;
  cb.dataset.value = opt;
  cb.id = `ctl-${key}-${opt}`;
  cb.addEventListener("change", onAnyChange);
  const span = document.createElement("span");
  span.textContent = opt;
  wrap.appendChild(cb);
  wrap.appendChild(span);
  return wrap;
}

function buildControl(c) {
  const row = document.createElement("div");
  row.className = "control-row";

  if (c.type === "multicheckbox") {
    row.classList.add("multicheckbox");
    const label = document.createElement("label");
    label.textContent = c.label;
    label.className = "ctl-label-full";
    row.appendChild(label);
    // Group by vendor + status for scannability when there are >10 options.
    if (c.key === "allowedGpus") {
      const groups = {};
      for (const opt of c.options) {
        const gpu = GPUS[opt];
        const key = `${gpu?.vendor || "other"} · ${gpu?.status || "?"}`;
        (groups[key] ??= []).push(opt);
      }
      for (const [groupName, opts] of Object.entries(groups)) {
        const wrap = document.createElement("div");
        wrap.className = "checkbox-group";
        const head = document.createElement("div");
        head.className = "checkbox-group-head";
        head.textContent = groupName;
        wrap.appendChild(head);
        for (const opt of opts) {
          wrap.appendChild(makeCheckOption(c.key, opt));
        }
        row.appendChild(wrap);
      }
      return row;
    }
    // Default flat grid for other multicheckbox controls.
    const grid = document.createElement("div");
    grid.className = "checkbox-grid";
    for (const opt of c.options) grid.appendChild(makeCheckOption(c.key, opt));
    row.appendChild(grid);
    return row;
  }

  const label = document.createElement("label");
  label.htmlFor = `ctl-${c.key}`;
  label.textContent = c.label;

  let input;
  if (c.type === "select") {
    input = document.createElement("select");
    for (const opt of c.options) {
      const o = document.createElement("option");
      o.value = opt;
      let display = opt;
      if (c.key === "gpuKey") display = GPUS[opt].label;
      else if (c.key === "kvScheme") display = KV_SCHEME_LABELS[opt] ?? opt;
      o.textContent = display;
      input.appendChild(o);
    }
  } else if (c.type === "checkbox") {
    input = document.createElement("input");
    input.type = "checkbox";
  } else {
    input = document.createElement("input");
    input.type = "number";
    if (c.min !== undefined) input.min = c.min;
    if (c.max !== undefined) input.max = c.max;
    if (c.step !== undefined) input.step = c.step;
  }
  input.id = `ctl-${c.key}`;
  input.dataset.key = c.key;
  input.addEventListener("change", onAnyChange);
  if (c.type !== "checkbox") {
    input.addEventListener("input", debounce(onAnyChange, 200));
  }
  row.appendChild(label);
  row.appendChild(input);
  return row;
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function applyInputsToDom(inputs) {
  for (const c of CONTROLS) {
    if (c.type === "multicheckbox") {
      const allowed = new Set(inputs[c.key] || c.options);
      for (const opt of c.options) {
        const el = document.getElementById(`ctl-${c.key}-${opt}`);
        if (!el) continue;
        el.checked = inputs[c.key] === null || inputs[c.key] === undefined
          ? true : allowed.has(opt);
      }
      continue;
    }
    const el = document.getElementById(`ctl-${c.key}`);
    if (!el) continue;
    if (c.type === "checkbox") el.checked = !!inputs[c.key];
    else el.value = inputs[c.key] ?? "";
  }
  applyPortfolioToDom(inputs.portfolio);
}

function readInputsFromDom() {
  const inputs = { ...currentInputs };
  for (const c of CONTROLS) {
    if (c.type === "multicheckbox") {
      const checked = [];
      for (const opt of c.options) {
        const el = document.getElementById(`ctl-${c.key}-${opt}`);
        if (el && el.checked) checked.push(opt);
      }
      inputs[c.key] = checked.length === c.options.length ? null : checked;
      continue;
    }
    const el = document.getElementById(`ctl-${c.key}`);
    if (!el) continue;
    if (c.type === "select") inputs[c.key] = el.value;
    else if (c.type === "checkbox") inputs[c.key] = el.checked;
    else inputs[c.key] = parseFloat(el.value);
  }
  inputs.portfolio = readPortfolioFromDom();
  return inputs;
}

function buildScenarioPicker() {
  const sel = document.getElementById("scenario-picker");
  sel.innerHTML = "";
  for (const [k, s] of Object.entries(SCENARIOS)) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = s.label;
    sel.appendChild(o);
  }
  sel.value = DEFAULT_SCENARIO;
  sel.addEventListener("change", () => {
    loadScenario(sel.value);
  });
}

function loadScenario(key) {
  currentInputs = { ...SCENARIOS[key].inputs };
  applyInputsToDom(currentInputs);
  document.getElementById("scenario-description").textContent = SCENARIOS[key].description;
  recompute();
}

function onAnyChange() {
  const inputs = readInputsFromDom();
  const sum = (inputs.edgeFacilityShare ?? 0)
            + (inputs.regionalFacilityShare ?? 0)
            + (inputs.centralFacilityShare ?? 0);
  if (sum > 0 && Math.abs(sum - 1) > 0.01) {
    document.getElementById("share-warn").textContent =
      `⚠ Facility shares sum to ${sum.toFixed(2)} (should be 1.0)`;
  } else {
    document.getElementById("share-warn").textContent = "";
  }
  currentInputs = inputs;
  recompute();
}

function onPortfolioChange() {
  // Same flow but lets us add portfolio-specific logic later (e.g. validation).
  onAnyChange();
}

function setKpi(id, v) {
  document.getElementById(id).textContent = v;
}

function recompute() {
  const t0 = performance.now();
  const plan = buildPlan(currentInputs);
  const elapsed = performance.now() - t0;

  setKpi("kpi-peak-gpus", fmt(plan.peakGpus));
  setKpi("kpi-peak-mw", plan.peakMw.toFixed(1) + " MW");
  setKpi("kpi-peak-facilities", plan.peakFacilities.toString());
  setKpi("kpi-tco", "$" + fmt(plan.totalDiscountedTco));
  const lastYear = plan.yearly[plan.yearly.length - 1];
  setKpi("kpi-internal-price", "$" + lastYear.internalDollarsPerMtok.toFixed(2) + "/Mtok");
  setKpi("kpi-fabric", lastYear.minScaleUpGpus + " GPUs");

  // Workload-portfolio frame
  renderDomainStack("chart-domain", plan);
  renderFp64Split("chart-fp64-split", plan);
  renderBoundHeatmap("chart-bound-heatmap", plan);
  renderSkuMix("chart-sku-mix", plan);

  // Chip-mix & spend frame
  renderFleetByWorkload("chart-fleet", plan);
  renderTcoStack("chart-tco", plan);
  renderBuildVsBuy("chart-buybuild", plan);

  // Latency & DC frame
  renderFacilities("chart-facilities", plan);
  renderPower("chart-power", plan);
  renderScaleUp("chart-scaleup", plan);
  const frontier = latencyCostFrontier(currentInputs, 0);
  renderLatencyFrontier("chart-latency", frontier);

  // Long-term & surge frame
  const fan = demandFan(currentInputs);
  renderDemandFan("chart-fan", fan);

  // Scaling-case fan (latent uncertainty across the portfolio).
  // Slightly expensive (3 buildPlan calls); defer to next tick.
  setTimeout(() => {
    const sFan = scalingCaseFan(currentInputs);
    renderScalingCaseFan("chart-scaling-fan", sFan);
  }, 0);

  setTimeout(() => {
    const tor = tornadoBattery(currentInputs, plan.totalDiscountedTco);
    renderTornado("chart-tornado", tor, plan.totalDiscountedTco);
  }, 0);

  document.getElementById("compute-time").textContent =
    `recomputed in ${elapsed.toFixed(0)} ms`;

  updateNarrative(plan);
}

function updateNarrative(plan) {
  const last = plan.yearly[plan.yearly.length - 1];
  const first = plan.yearly[0];
  const fleetGrowth = last.totalGpus / Math.max(1, first.totalGpus);
  const tcoMillions = (plan.totalDiscountedTco / 1e6).toFixed(0);
  const buildBeatsRent = last.internalDollarsPerMtok < last.externalDollarsPerMtok;
  const fabricNeeded = last.minScaleUpGpus;
  const scaleUpDomain = plan.gpu?.scaleup_domain || 8;
  const fabricStress = fabricNeeded > scaleUpDomain;

  const skuMix = Object.entries(last.gpusBySku || {})
    .sort((a, b) => b[1] - a[1])
    .map(([sku, count]) => `${sku} (${fmt(Math.ceil(count))})`)
    .join(", ");
  const skuCount = Object.keys(last.gpusBySku || {}).length;

  // Domain mix: workload domains contributing GPU-years
  const domainMix = Object.entries(last.gpuYearsByDomain || {})
    .filter(([_, gy]) => gy > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([d, gy]) => `${DOMAINS[d]?.label || d} (${fmt(Math.round(gy))} GPU-yr)`)
    .join(", ");

  const intP = last.allocations?.interactive_inference;
  const intDecode = intP?.decode;
  const intPrefill = intP?.prefill;
  const specMul = intP?.specMul;

  const lines = [
    `Over ${plan.yearly.length} years, the fleet grows ${fleetGrowth.toFixed(1)}× to ${fmt(last.totalGpus)} GPUs across ${last.numFacilities} facility(s) drawing ${last.totalKw / 1000 | 0} MW peak.`,
    `Total discounted TCO: $${tcoMillions}M. Fleet mix: ${skuMix || "—"} (${skuCount} distinct SKU${skuCount === 1 ? "" : "s"}).`,
    domainMix
      ? `Workload domains: ${domainMix}.`
      : `(No workload demand specified.)`,
    intDecode?.sku && intPrefill?.sku
      ? `Interactive serving: prefill on ${intPrefill.sku} (TP=${intPrefill.tp}, batch=${intPrefill.batch}, ${intPrefill.latencyMs?.toFixed(0)}ms total), decode on ${intDecode.sku} (TP=${intDecode.tp}, batch=${intDecode.batch}, ${intDecode.latencyMs?.toFixed(0)}ms/tok).`
      : `Interactive LLM serving disabled or infeasible.`,
    specMul && specMul > 1.05
      ? `Speculative decoding contributing ${specMul.toFixed(2)}× decode throughput speedup.`
      : `Speculative decoding off or contributing <5% (acceptance prob too low or γ=0).`,
    last.tokensServed > 0 && buildBeatsRent
      ? `By Y${last.year}, internal $${last.internalDollarsPerMtok.toFixed(2)}/Mtok beats the external $${last.externalDollarsPerMtok.toFixed(2)}/Mtok benchmark.`
      : last.tokensServed > 0
      ? `By Y${last.year}, external rental ($${last.externalDollarsPerMtok.toFixed(2)}/Mtok) is cheaper than internal ($${last.internalDollarsPerMtok.toFixed(2)}/Mtok). Reconsider build vs rent.`
      : `Build-vs-rent comparison N/A (no LLM token-serving workloads in this portfolio).`,
    fabricStress
      ? `⚠ The frontier model (${last.yearParamsB?.toFixed?.(0) ?? "—"}B params) requires ${fabricNeeded} coherent-fabric GPUs, exceeding the ${plan.gpu?.label || plan.headlineSku} scale-up domain of ${scaleUpDomain}.`
      : `LLM frontier model fits in a ${fabricNeeded}-GPU scale-up domain on ${plan.gpu?.label || plan.headlineSku}.`,
    plan.anyShortfall
      ? `⚠ Site capacity insufficient in ≥1 year of the plan; consider neocloud overflow or earlier site builds (lead times 2–3 years).`
      : `Site capacity adequate across all regions for the entire horizon.`,
  ];
  document.getElementById("narrative").innerHTML = lines.map((l) => `<p>${l}</p>`).join("");
}

export function init() {
  buildKpiRow();
  buildHatPills();
  buildControlsDom();
  buildScenarioPicker();
  setHat(DEFAULT_HAT);
  loadScenario(DEFAULT_SCENARIO);
  document.getElementById("recompute-btn").addEventListener("click", recompute);
}
