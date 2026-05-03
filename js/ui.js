// Wires DOM controls to the model and charts.

import { GPUS } from "./gpus.js";
import { SCENARIOS, DEFAULT_SCENARIO } from "./scenarios.js";
import {
  buildPlan,
  tornadoBattery,
  demandFan,
  latencyCostFrontier,
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
  fmt,
} from "./charts.js";

// Spec for every input. type=number|select|range, key matches model input field.
const CONTROLS = [
  { group: "Planning horizon", key: "horizonYears", label: "Years", type: "number", min: 1, max: 10, step: 1 },
  { group: "Planning horizon", key: "discountRate", label: "Discount rate", type: "number", min: 0, max: 0.25, step: 0.01 },
  { group: "Planning horizon", key: "refreshYears", label: "Refresh cycle (yrs)", type: "number", min: 2, max: 7, step: 1 },

  { group: "Hardware", key: "gpuKey", label: "GPU SKU", type: "select", options: Object.keys(GPUS) },
  { group: "Hardware", key: "quantization", label: "Quantization", type: "select", options: ["fp16", "fp8", "int4"] },
  { group: "Hardware", key: "tpDegree", label: "Tensor-parallel degree", type: "number", min: 1, max: 64, step: 1 },
  { group: "Hardware", key: "networkTier", label: "Network fabric tier", type: "select", options: ["scale_up_only", "small_pod", "large_pod", "full_fabric"] },
  { group: "Hardware", key: "pue", label: "Datacenter PUE", type: "number", min: 1.05, max: 2.0, step: 0.01 },
  { group: "Hardware", key: "electricityPrice", label: "Electricity ($/kWh)", type: "number", min: 0.02, max: 0.30, step: 0.005 },

  { group: "Model", key: "paramsB", label: "Model size (B params)", type: "number", min: 1, max: 5000, step: 1 },
  { group: "Model", key: "activeFrac", label: "Active param fraction (MoE)", type: "number", min: 0.05, max: 1.0, step: 0.05 },
  { group: "Model", key: "contextLen", label: "Avg context length (tokens)", type: "number", min: 1000, max: 200_000, step: 1000 },
  { group: "Model", key: "kvBytesPerToken", label: "KV cache bytes/token", type: "number", min: 10_000, max: 1_000_000, step: 10_000 },

  { group: "Demand", key: "interactiveTokensPerDay", label: "Interactive tokens/day", type: "number", min: 0, max: 1e12, step: 1e8 },
  { group: "Demand", key: "batchTokensPerDay", label: "Batch tokens/day", type: "number", min: 0, max: 1e12, step: 1e8 },
  { group: "Demand", key: "rlTokensPerDay", label: "RL rollout tokens/day", type: "number", min: 0, max: 1e12, step: 1e8 },
  { group: "Demand", key: "pretrainFlopsPerYear", label: "Pretraining FLOPs/year", type: "number", min: 0, max: 1e27, step: 1e22 },
  { group: "Demand", key: "finetuneFlopsPerYear", label: "Finetune FLOPs/year", type: "number", min: 0, max: 1e26, step: 1e21 },
  { group: "Demand", key: "demandGrowth", label: "Demand YoY growth", type: "number", min: 0, max: 3.0, step: 0.05 },
  { group: "Demand", key: "modelGrowth", label: "Model size YoY growth", type: "number", min: 0, max: 2.0, step: 0.05 },
  { group: "Demand", key: "algEfficiency", label: "Alg efficiency YoY gain", type: "number", min: 0, max: 2.0, step: 0.05 },

  { group: "Operations", key: "mfu", label: "Training MFU", type: "number", min: 0.05, max: 0.7, step: 0.01 },
  { group: "Operations", key: "utilization", label: "Inference fleet utilization", type: "number", min: 0.2, max: 0.95, step: 0.01 },
  { group: "Operations", key: "headroomFactor", label: "Capacity headroom factor", type: "number", min: 1.0, max: 2.5, step: 0.05 },

  { group: "Strategy", key: "rentalShare", label: "Rental share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "edgeFacilityShare", label: "Edge facility GPU share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "regionalFacilityShare", label: "Regional facility GPU share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "centralFacilityShare", label: "Central facility GPU share", type: "number", min: 0, max: 1, step: 0.05 },
  { group: "Strategy", key: "facilityMaxMw", label: "Max MW per facility", type: "number", min: 5, max: 500, step: 5 },

  { group: "External benchmark", key: "externalInitialPrice", label: "External $/Mtok today", type: "number", min: 0.1, max: 100, step: 0.1 },
  { group: "External benchmark", key: "externalDeclineMult", label: "External price decline (x/yr)", type: "number", min: 1, max: 50, step: 0.5 },
];

let currentInputs = {};

function buildControlsDom() {
  const container = document.getElementById("controls");
  const groups = {};
  for (const c of CONTROLS) {
    if (!groups[c.group]) groups[c.group] = [];
    groups[c.group].push(c);
  }
  container.innerHTML = "";
  for (const [groupName, controls] of Object.entries(groups)) {
    const section = document.createElement("section");
    section.className = "control-group";
    section.innerHTML = `<h3>${groupName}</h3>`;
    for (const c of controls) {
      const row = document.createElement("div");
      row.className = "control-row";
      const label = document.createElement("label");
      label.htmlFor = `ctl-${c.key}`;
      label.textContent = c.label;

      let input;
      if (c.type === "select") {
        input = document.createElement("select");
        for (const opt of c.options) {
          const o = document.createElement("option");
          o.value = opt;
          o.textContent = c.key === "gpuKey" ? GPUS[opt].label : opt;
          input.appendChild(o);
        }
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
      input.addEventListener("input", debounce(onAnyChange, 200));
      row.appendChild(label);
      row.appendChild(input);
      section.appendChild(row);
    }
    container.appendChild(section);
  }
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function applyInputsToDom(inputs) {
  for (const c of CONTROLS) {
    const el = document.getElementById(`ctl-${c.key}`);
    if (!el) continue;
    el.value = inputs[c.key];
  }
}

function readInputsFromDom() {
  const inputs = { ...currentInputs };
  for (const c of CONTROLS) {
    const el = document.getElementById(`ctl-${c.key}`);
    if (!el) continue;
    if (c.type === "select") inputs[c.key] = el.value;
    else inputs[c.key] = parseFloat(el.value);
  }
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
  // Clamp the three facility shares to sum to 1 (warn if not).
  const inputs = readInputsFromDom();
  const sum = inputs.edgeFacilityShare + inputs.regionalFacilityShare + inputs.centralFacilityShare;
  if (sum > 0 && Math.abs(sum - 1) > 0.01) {
    document.getElementById("share-warn").textContent =
      `⚠ Facility shares sum to ${sum.toFixed(2)} (should be 1.0)`;
  } else {
    document.getElementById("share-warn").textContent = "";
  }
  currentInputs = inputs;
  recompute();
}

function setKpi(id, v) {
  document.getElementById(id).textContent = v;
}

function recompute() {
  const t0 = performance.now();
  const plan = buildPlan(currentInputs);
  const elapsed = performance.now() - t0;

  // KPI tiles
  setKpi("kpi-peak-gpus", fmt(plan.peakGpus));
  setKpi("kpi-peak-mw", plan.peakMw.toFixed(1) + " MW");
  setKpi("kpi-peak-facilities", plan.peakFacilities.toString());
  setKpi("kpi-tco", "$" + fmt(plan.totalDiscountedTco));
  const lastYear = plan.yearly[plan.yearly.length - 1];
  setKpi("kpi-internal-price", "$" + lastYear.internalDollarsPerMtok.toFixed(2) + "/Mtok");
  setKpi("kpi-fabric", lastYear.minScaleUpGpus + " GPUs");

  // Charts
  renderFleetByWorkload("chart-fleet", plan);
  renderFacilities("chart-facilities", plan);
  renderPower("chart-power", plan);
  renderTcoStack("chart-tco", plan);
  renderBuildVsBuy("chart-buybuild", plan);
  renderScaleUp("chart-scaleup", plan);

  const fan = demandFan(currentInputs);
  renderDemandFan("chart-fan", fan);

  const frontier = latencyCostFrontier(currentInputs, 0);
  renderLatencyFrontier("chart-latency", frontier);

  // Tornado is more expensive (10 perturbations); run async-ish.
  setTimeout(() => {
    const tor = tornadoBattery(currentInputs, plan.totalDiscountedTco);
    renderTornado("chart-tornado", tor, plan.totalDiscountedTco);
  }, 0);

  document.getElementById("compute-time").textContent =
    `recomputed in ${elapsed.toFixed(0)} ms`;

  // Update narrative
  updateNarrative(plan);
}

function updateNarrative(plan) {
  const last = plan.yearly[plan.yearly.length - 1];
  const first = plan.yearly[0];
  const fleetGrowth = last.totalGpus / Math.max(1, first.totalGpus);
  const tcoMillions = (plan.totalDiscountedTco / 1e6).toFixed(0);
  const buildBeatsRent = last.internalDollarsPerMtok < last.externalDollarsPerMtok;
  const fabricNeeded = last.minScaleUpGpus;
  const scaleUpDomain = plan.gpu.scaleup_domain || 8;
  const fabricStress = fabricNeeded > scaleUpDomain;
  const lines = [
    `Over ${plan.yearly.length} years, the fleet grows ${fleetGrowth.toFixed(1)}× to ${fmt(last.totalGpus)} GPUs across ${last.numFacilities} facility(s) drawing ${last.totalKw / 1000 | 0} MW peak.`,
    `Total discounted TCO: $${tcoMillions}M (${plan.gpu.label}, ${(last.totalKw / 1000).toFixed(1)} MW year ${plan.yearly.length - 1}).`,
    buildBeatsRent
      ? `By Y${last.year}, internal $${last.internalDollarsPerMtok.toFixed(2)}/Mtok beats the external $${last.externalDollarsPerMtok.toFixed(2)}/Mtok benchmark.`
      : `By Y${last.year}, external rental ($${last.externalDollarsPerMtok.toFixed(2)}/Mtok) is cheaper than internal ($${last.internalDollarsPerMtok.toFixed(2)}/Mtok). Reconsider build vs rent.`,
    fabricStress
      ? `⚠ The frontier model (${last.yearParamsB.toFixed(0)}B params) requires ${fabricNeeded} coherent-fabric GPUs, exceeding the ${plan.gpu.label} scale-up domain of ${scaleUpDomain}. Plan for next-gen hardware or aggressive quantization.`
      : `The frontier model (${last.yearParamsB.toFixed(0)}B params) fits in a ${fabricNeeded}-GPU scale-up domain.`,
  ];
  document.getElementById("narrative").innerHTML = lines.map((l) => `<p>${l}</p>`).join("");
}

export function init() {
  buildControlsDom();
  buildScenarioPicker();
  loadScenario(DEFAULT_SCENARIO);
  document.getElementById("recompute-btn").addEventListener("click", recompute);
}
