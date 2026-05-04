import { GPUS } from './gpus.js';
import { SCENARIOS, DEFAULT_SCENARIO } from './scenarios.js';
import { buildPlan, tornadoBattery, demandFan, latencyCostFrontier, scalingCaseFan } from './model.js';
import {
  renderFleetByWorkload, renderFacilities, renderPower,
  renderTcoStack, renderBuildVsBuy, renderLatencyFrontier,
  renderTornado, renderDemandFan, renderScaleUp,
  renderDomainStack, renderScalingCaseFan, renderFp64Split,
  renderBoundHeatmap, renderSkuMix, fmt
} from './charts.js';
import { KV_SCHEMES, KV_SCHEME_LABELS } from './kv.js';
import { WORKLOAD_CLASSES, DOMAINS, workloadsByDomain, SCALING_CASES } from './workload_classes.js';
import { makePortfolioItem } from './portfolio.js';
import { HATS, DEFAULT_HAT, KPI_META } from './hats.js';

// ─── State ───────────────────────────────────────────────────────────────────
let currentInputs = {};
let currentHat = DEFAULT_HAT;
let activeScenario = DEFAULT_SCENARIO;

// ─── Gateway ─────────────────────────────────────────────────────────────────
function buildGateway() {
  const grid = document.getElementById('gateway-personas');
  for (const [id, hat] of Object.entries(HATS)) {
    if (id === 'all') continue;
    const card = document.createElement('div');
    card.className = 'persona-card';
    card.dataset.hat = id;
    card.innerHTML = `
      <div class="persona-icon">${hat.icon}</div>
      <div class="persona-name">${hat.short}</div>
      <div class="persona-desc">${hat.blurb.split('.')[0]}.</div>
    `;
    card.addEventListener('click', () => {
      document.querySelectorAll('.persona-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
    });
    grid.appendChild(card);
  }
  document.getElementById('gateway-enter').addEventListener('click', () => {
    const sel = document.querySelector('.persona-card.selected');
    openDashboard(sel ? sel.dataset.hat : 'all');
  });
  document.getElementById('gateway-skip').addEventListener('click', () => openDashboard('all'));
}

function openDashboard(hatId) {
  const gw = document.getElementById('gateway');
  gw.classList.add('leaving');
  setTimeout(() => {
    gw.classList.add('gone');
    document.getElementById('app').style.display = '';
    setHat(hatId);
    recompute();
  }, 380);
}

// ─── Sidebar tabs ─────────────────────────────────────────────────────────────
function initSidebarTabs() {
  document.querySelectorAll('.sidebar-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.sidebar-pane').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`pane-${tab.dataset.tab}`).classList.add('active');
    });
  });
}

// ─── Hat system ──────────────────────────────────────────────────────────────
function buildHatPillsHeader() {
  const wrap = document.getElementById('hat-pills-header');
  wrap.innerHTML = '';
  for (const [id, hat] of Object.entries(HATS)) {
    const btn = document.createElement('button');
    btn.className = 'hat-pill-h';
    btn.dataset.hat = id;
    btn.innerHTML = `<span class="pill-icon">${hat.icon}</span>${hat.short}`;
    btn.title = hat.label;
    btn.addEventListener('click', () => setHat(id));
    wrap.appendChild(btn);
  }
}

function setHat(hatId) {
  currentHat = hatId;
  document.querySelectorAll('.hat-pill-h').forEach(p =>
    p.classList.toggle('active', p.dataset.hat === hatId));
  document.querySelectorAll('.hat-badge').forEach(b =>
    b.classList.toggle('active', b.dataset.hat === hatId));

  const hat = HATS[hatId];
  document.querySelectorAll('.kpi').forEach(el => {
    const isKey = hat.keyKpis?.includes(el.dataset.kpiId);
    el.classList.toggle('primary-for-hat', hatId !== 'all' && isKey);
    el.classList.toggle('secondary-for-hat', hatId !== 'all' && !isKey);
  });
  document.querySelectorAll('.chart-frame').forEach(el => {
    const isPrimary = hat.primaryFrames?.includes(el.dataset.frame);
    el.classList.toggle('primary-for-hat', isPrimary);
    el.classList.toggle('secondary-for-hat', hatId !== 'all' && !isPrimary);
  });
  const ins = document.getElementById('hat-insight');
  if (hatId === 'all' || !hat.questions) {
    ins.classList.remove('visible');
  } else {
    ins.classList.add('visible');
    document.getElementById('hi-icon').textContent = hat.icon;
    document.getElementById('hi-who').textContent = hat.label;
    document.getElementById('hi-blurb').textContent = hat.blurb;
    const ql = document.getElementById('hi-questions');
    ql.innerHTML = hat.questions.map(q => `<li>${q}</li>`).join('');
  }
}

// ─── KPI row ─────────────────────────────────────────────────────────────────
function buildKpiRow() {
  const row = document.getElementById('kpi-row');
  row.innerHTML = '';
  const order = ['kpi-peak-gpus','kpi-peak-mw','kpi-peak-facilities','kpi-tco','kpi-internal-price','kpi-fabric'];
  for (const id of order) {
    const meta = KPI_META[id];
    if (!meta) continue;
    const tile = document.createElement('div');
    tile.className = 'kpi';
    tile.dataset.kpiId = id;
    tile.innerHTML = `
      <div class="kpi-label">
        ${meta.label}
        <span class="kpi-info" title="${meta.explainer.replace(/"/g,'&quot;')}">i</span>
      </div>
      <div class="kpi-value" id="${id}">—</div>
      <div class="kpi-hats">${meta.hats.map(h => `<span class="hat-badge" data-hat="${h}">${HATS[h].short}</span>`).join('')}</div>
    `;
    row.appendChild(tile);
  }
}

function setKpi(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('updating');
  el.textContent = val;
  setTimeout(() => el.classList.remove('updating'), 500);
}

// ─── Scenario cards ──────────────────────────────────────────────────────────
function buildScenarioCards() {
  const wrap = document.getElementById('scenario-cards');
  wrap.innerHTML = '';
  for (const [key, s] of Object.entries(SCENARIOS)) {
    const card = document.createElement('div');
    card.className = 'scenario-card' + (key === activeScenario ? ' active' : '');
    card.dataset.scenario = key;
    card.innerHTML = `<div class="scenario-card-name">${s.label}</div><div class="scenario-card-desc">${s.description}</div>`;
    card.addEventListener('click', () => loadScenario(key));
    wrap.appendChild(card);
  }
}

function loadScenario(key) {
  activeScenario = key;
  currentInputs = { ...SCENARIOS[key].inputs };
  applyInputsToDom(currentInputs);
  document.querySelectorAll('.scenario-card').forEach(c =>
    c.classList.toggle('active', c.dataset.scenario === key));
  recompute();
}

// ─── Portfolio editor ────────────────────────────────────────────────────────
function buildPortfolioEditor() {
  const wrap = document.getElementById('portfolio-editor');
  wrap.innerHTML = '';
  const byDomain = workloadsByDomain();
  for (const [domain, classes] of Object.entries(byDomain)) {
    const dom = DOMAINS[domain];
    const block = document.createElement('div');
    block.className = 'domain-block';
    block.dataset.domain = domain;
    block.innerHTML = `
      <div class="domain-header">
        <span class="domain-dot" style="background:${dom.color}"></span>
        <span>${dom.label}</span>
        <span class="domain-chevron">▾</span>
      </div>
      <div class="domain-rows"></div>
    `;
    block.querySelector('.domain-header').addEventListener('click', () =>
      block.classList.toggle('collapsed'));
    const rows = block.querySelector('.domain-rows');
    for (const cls of classes) {
      rows.appendChild(buildPfRow(cls));
    }
    wrap.appendChild(block);
  }
}

function buildPfRow(cls) {
  const row = document.createElement('div');
  row.className = 'pf-row';
  row.dataset.classId = cls.id;
  const cases = cls.domain === 'llm' ? SCALING_CASES : SCALING_CASES.filter(s => s !== 'fitted');
  row.innerHTML = `
    <div class="pf-row-top">
      <input type="checkbox" class="pf-enable" data-class-id="${cls.id}" />
      <label for="pf-cb-${cls.id}">${cls.label}</label>
    </div>
    <div class="pf-inputs">
      <input type="number" class="pf-demand" data-class-id="${cls.id}" min="0" step="any" placeholder="${cls.default_demand?.label || 'units/yr'}" />
      <select class="pf-scaling" data-class-id="${cls.id}">
        ${cases.map(s => `<option value="${s}">${s}</option>`).join('')}
      </select>
    </div>
    <div class="pf-meta">${cls.compute_pattern} · ${cls.bound_default || ''}${cls.fp64_required ? ' · FP64' : ''}</div>
  `;
  row.querySelector('.pf-enable').addEventListener('change', onPortfolioChange);
  row.querySelector('.pf-demand').addEventListener('input', debounce(onPortfolioChange, 250));
  row.querySelector('.pf-scaling').addEventListener('change', onPortfolioChange);
  return row;
}

function applyPortfolioToDom(portfolio) {
  for (const cls of Object.values(WORKLOAD_CLASSES)) {
    const en = document.querySelector(`.pf-enable[data-class-id="${cls.id}"]`);
    const dm = document.querySelector(`.pf-demand[data-class-id="${cls.id}"]`);
    const sc = document.querySelector(`.pf-scaling[data-class-id="${cls.id}"]`);
    if (!en) continue;
    en.checked = false; dm.value = ''; sc.value = 'median';
  }
  if (!Array.isArray(portfolio)) return;
  for (const item of portfolio) {
    const en = document.querySelector(`.pf-enable[data-class-id="${item.classId}"]`);
    const dm = document.querySelector(`.pf-demand[data-class-id="${item.classId}"]`);
    const sc = document.querySelector(`.pf-scaling[data-class-id="${item.classId}"]`);
    if (!en) continue;
    en.checked = item.enabled !== false;
    dm.value = item.unitsPerYear;
    sc.value = item.scalingCase || 'median';
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
    items.push(makePortfolioItem(cls.id, { unitsPerYear: val, scalingCase: sc.value }));
  }
  return items;
}

// ─── Parameter controls ──────────────────────────────────────────────────────
const PARAM_GROUPS = [
  {
    key: 'horizon', label: 'Planning horizon', items: [
      { key: 'horizonYears', label: 'Years', type: 'number', min:1, max:10, step:1 },
      { key: 'discountRate', label: 'Discount rate', type: 'number', min:0, max:0.25, step:0.01 },
      { key: 'refreshYears', label: 'Refresh cycle (yrs)', type: 'number', min:2, max:7, step:1 },
    ]
  },
  {
    key: 'hardware', label: 'Hardware', items: [
      { key: 'gpuKey', label: 'Headline GPU SKU', type: 'select', options: Object.keys(GPUS) },
      { key: 'allowedGpus', label: 'Allowed SKUs', type: 'multicheckbox', options: Object.keys(GPUS) },
      { key: 'quantization', label: 'Quantization (LLM)', type: 'select', options: ['fp16','fp8','int4'] },
      { key: 'networkTier', label: 'Network fabric tier', type: 'select', options: ['scale_up_only','small_pod','large_pod','full_fabric'] },
      { key: 'pue', label: 'Datacenter PUE', type: 'number', min:1.05, max:2.0, step:0.01 },
      { key: 'electricityPrice', label: 'Electricity ($/kWh)', type: 'number', min:0.02, max:0.30, step:0.005 },
    ]
  },
  {
    key: 'model', label: 'LLM model', items: [
      { key: 'paramsB', label: 'Size (B params)', type: 'number', min:1, max:5000, step:1 },
      { key: 'activeFrac', label: 'Active frac (MoE)', type: 'number', min:0.05, max:1.0, step:0.05 },
      { key: 'contextLen', label: 'Avg input length (tok)', type: 'number', min:1000, max:200000, step:1000 },
      { key: 'outputLen', label: 'Avg output length (tok)', type: 'number', min:50, max:10000, step:50 },
    ]
  },
  {
    key: 'runtime', label: 'Architecture & runtime', items: [
      { key: 'kvScheme', label: 'KV-cache scheme', type: 'select', options: KV_SCHEMES },
      { key: 'enableSpecDec', label: 'Speculative decoding', type: 'checkbox' },
      { key: 'specDecAcceptanceProb', label: 'Spec-dec acceptance', type: 'number', min:0, max:0.99, step:0.05 },
      { key: 'specDecGammaMax', label: 'Spec-dec γ (draft toks)', type: 'number', min:1, max:10, step:1 },
      { key: 'enableQueueing', label: 'Queueing-derived util cap', type: 'checkbox' },
    ]
  },
  {
    key: 'ops', label: 'Operations', items: [
      { key: 'mfu', label: 'Training MFU', type: 'number', min:0.05, max:0.7, step:0.01 },
      { key: 'utilization', label: 'Inference fleet util cap', type: 'number', min:0.2, max:0.95, step:0.01 },
      { key: 'headroomFactor', label: 'Capacity headroom', type: 'number', min:1.0, max:2.5, step:0.05 },
      { key: 'demandGrowth', label: 'Default YoY demand growth', type: 'number', min:0, max:3.0, step:0.05 },
      { key: 'modelGrowth', label: 'LLM model size YoY growth', type: 'number', min:0, max:2.0, step:0.05 },
      { key: 'algEfficiency', label: 'Alg efficiency YoY gain', type: 'number', min:0, max:2.0, step:0.05 },
    ]
  },
  {
    key: 'strategy', label: 'Strategy', items: [
      { key: 'rentalShare', label: 'Rental share', type: 'number', min:0, max:1, step:0.05 },
      { key: 'edgeFacilityShare', label: 'Edge facility GPU share', type: 'number', min:0, max:1, step:0.05 },
      { key: 'regionalFacilityShare', label: 'Regional GPU share', type: 'number', min:0, max:1, step:0.05 },
      { key: 'centralFacilityShare', label: 'Central GPU share', type: 'number', min:0, max:1, step:0.05 },
      { key: 'facilityMaxMw', label: 'Max MW per facility', type: 'number', min:5, max:500, step:5 },
    ]
  },
  {
    key: 'benchmark', label: 'External benchmark', items: [
      { key: 'externalInitialPrice', label: 'External $/Mtok today', type: 'number', min:0.1, max:100, step:0.1 },
      { key: 'externalDeclineMult', label: 'External price decline (x/yr)', type: 'number', min:1, max:50, step:0.5 },
    ]
  },
];

const ALL_PARAM_CONTROLS = PARAM_GROUPS.flatMap(g => g.items);

function buildParamsControls() {
  const wrap = document.getElementById('params-controls');
  wrap.innerHTML = '';
  for (const group of PARAM_GROUPS) {
    const section = document.createElement('div');
    section.className = 'ctrl-section';
    section.dataset.group = group.key;
    section.innerHTML = `
      <div class="ctrl-section-header">
        <span class="ctrl-section-title">${group.label}</span>
        <span class="ctrl-chevron">▾</span>
      </div>
      <div class="ctrl-body"></div>
    `;
    section.querySelector('.ctrl-section-header').addEventListener('click', () =>
      section.classList.toggle('collapsed'));
    const body = section.querySelector('.ctrl-body');
    for (const c of group.items) {
      body.appendChild(buildParamControl(c));
    }
    wrap.appendChild(section);
  }
}

function buildParamControl(c) {
  if (c.type === 'multicheckbox') {
    const wrap = document.createElement('div');
    wrap.className = 'ctrl-row full-width multicheckbox-section';
    wrap.innerHTML = `<div class="ctrl-label" style="margin-bottom:4px">${c.label}</div>`;
    const groups = {};
    for (const opt of c.options) {
      const gpu = GPUS[opt];
      const gk = `${gpu?.vendor || 'other'} · ${gpu?.status || '?'}`;
      if (!groups[gk]) groups[gk] = [];
      groups[gk].push(opt);
    }
    for (const [gname, opts] of Object.entries(groups)) {
      const groupWrap = document.createElement('div');
      groupWrap.innerHTML = `<div class="multicheckbox-group-label">${gname}</div><div class="checkbox-mini-grid"></div>`;
      const grid = groupWrap.querySelector('.checkbox-mini-grid');
      for (const opt of opts) {
        const label = document.createElement('label');
        label.className = 'check-mini';
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.dataset.key = c.key; cb.dataset.value = opt;
        cb.id = `ctl-${c.key}-${opt}`;
        cb.addEventListener('change', onAnyChange);
        const span = document.createElement('span');
        span.textContent = opt.replace(/_/g,' ');
        label.appendChild(cb); label.appendChild(span);
        grid.appendChild(label);
      }
      wrap.appendChild(groupWrap);
    }
    return wrap;
  }
  const row = document.createElement('div');
  row.className = 'ctrl-row';
  const label = document.createElement('label');
  label.htmlFor = `ctl-${c.key}`;
  label.className = 'ctrl-label';
  label.textContent = c.label;
  let input;
  if (c.type === 'select') {
    input = document.createElement('select');
    for (const opt of c.options) {
      const o = document.createElement('option');
      o.value = opt;
      if (c.key === 'gpuKey') o.textContent = GPUS[opt]?.label || opt;
      else if (c.key === 'kvScheme') o.textContent = KV_SCHEME_LABELS?.[opt] || opt;
      else o.textContent = opt;
      input.appendChild(o);
    }
  } else if (c.type === 'checkbox') {
    input = document.createElement('input');
    input.type = 'checkbox';
  } else {
    input = document.createElement('input');
    input.type = 'number';
    if (c.min !== undefined) input.min = c.min;
    if (c.max !== undefined) input.max = c.max;
    if (c.step !== undefined) input.step = c.step;
  }
  input.id = `ctl-${c.key}`;
  input.dataset.key = c.key;
  input.addEventListener('change', onAnyChange);
  if (c.type !== 'checkbox') input.addEventListener('input', debounce(onAnyChange, 250));
  row.appendChild(label);
  row.appendChild(input);
  return row;
}

function applyInputsToDom(inputs) {
  for (const c of ALL_PARAM_CONTROLS) {
    if (c.type === 'multicheckbox') {
      for (const opt of c.options) {
        const el = document.getElementById(`ctl-${c.key}-${opt}`);
        if (!el) continue;
        el.checked = !inputs[c.key] || inputs[c.key].includes(opt);
      }
      continue;
    }
    const el = document.getElementById(`ctl-${c.key}`);
    if (!el) continue;
    if (c.type === 'checkbox') el.checked = !!inputs[c.key];
    else el.value = inputs[c.key] ?? '';
  }
  applyPortfolioToDom(inputs.portfolio);
}

function readInputsFromDom() {
  const inputs = { ...currentInputs };
  for (const c of ALL_PARAM_CONTROLS) {
    if (c.type === 'multicheckbox') {
      const checked = c.options.filter(opt => {
        const el = document.getElementById(`ctl-${c.key}-${opt}`);
        return el && el.checked;
      });
      inputs[c.key] = checked.length === c.options.length ? null : checked;
      continue;
    }
    const el = document.getElementById(`ctl-${c.key}`);
    if (!el) continue;
    if (c.type === 'select') inputs[c.key] = el.value;
    else if (c.type === 'checkbox') inputs[c.key] = el.checked;
    else inputs[c.key] = parseFloat(el.value);
  }
  inputs.portfolio = readPortfolioFromDom();
  return inputs;
}

// ─── Event handlers ──────────────────────────────────────────────────────────
function onAnyChange() {
  const inputs = readInputsFromDom();
  const sum = (inputs.edgeFacilityShare || 0) + (inputs.regionalFacilityShare || 0) + (inputs.centralFacilityShare || 0);
  const warn = document.getElementById('share-warn');
  warn.textContent = (sum > 0 && Math.abs(sum - 1) > 0.01)
    ? `⚠ Facility shares sum to ${sum.toFixed(2)} (should be 1.0)` : '';
  currentInputs = inputs;
  recompute();
}
function onPortfolioChange() { onAnyChange(); }

// ─── Narrative ───────────────────────────────────────────────────────────────
function updateNarrative(plan) {
  const last = plan.yearly[plan.yearly.length - 1];
  const first = plan.yearly[0];
  const grow = last.totalGpus / Math.max(1, first.totalGpus);
  const tcoM = (plan.totalDiscountedTco / 1e6).toFixed(0);
  const buildBeats = last.internalDollarsPerMtok < last.externalDollarsPerMtok;
  const skuMix = Object.entries(last.gpusBySku || {}).sort((a,b) => b[1]-a[1])
    .map(([sku, count]) => `<strong>${sku}</strong> (${fmt(Math.ceil(count))})`).join(', ');
  const domainMix = Object.entries(last.gpuYearsByDomain || {})
    .filter(([,gy]) => gy > 0).sort((a,b) => b[1]-a[1])
    .map(([d, gy]) => `<strong>${DOMAINS[d]?.label || d}</strong> (${fmt(Math.round(gy))} GPU-yr)`).join(', ');
  const fabricNeeded = last.minScaleUpGpus;
  const scaleUpDomain = plan.gpu?.scaleup_domain || 8;
  const fabricStress = fabricNeeded > scaleUpDomain;
  const intP = last.allocations?.interactive_inference;
  const intDecode = intP?.decode;
  const intPrefill = intP?.prefill;

  const lines = [];
  lines.push(`Over ${plan.yearly.length} years, the fleet grows <strong>${grow.toFixed(1)}×</strong> to <strong>${fmt(last.totalGpus)} GPUs</strong> across <strong>${last.numFacilities} facilit${last.numFacilities === 1 ? 'y' : 'ies'}</strong> drawing <strong>${(last.totalKw/1000)|0} MW</strong> peak. Total discounted TCO: <strong>$${tcoM}M</strong>.`);
  if (skuMix) lines.push(`Fleet mix: ${skuMix}.`);
  if (domainMix) lines.push(`Dominant workload domains: ${domainMix}.`);
  if (intDecode?.sku && intPrefill?.sku) {
    lines.push(`Interactive LLM: prefill on <strong>${intPrefill.sku}</strong> (TP=${intPrefill.tp}, ${intPrefill.latencyMs?.toFixed(0)}ms), decode on <strong>${intDecode.sku}</strong> (TP=${intDecode.tp}, ${intDecode.latencyMs?.toFixed(0)}ms/tok).`);
  }
  if (last.tokensServed > 0) {
    if (buildBeats) {
      lines.push(`<span class="good">✓ By Y${last.year}, internal <strong>$${last.internalDollarsPerMtok.toFixed(2)}/Mtok</strong> beats the external <strong>$${last.externalDollarsPerMtok.toFixed(2)}/Mtok</strong> benchmark — building is justified.</span>`);
    } else {
      lines.push(`<span class="bad">⚠ By Y${last.year}, external rental (<strong>$${last.externalDollarsPerMtok.toFixed(2)}/Mtok</strong>) is still cheaper than internal (<strong>$${last.internalDollarsPerMtok.toFixed(2)}/Mtok</strong>). Reconsider build vs rent.</span>`);
    }
  }
  if (fabricStress) {
    lines.push(`<span class="warn">⚠ Frontier model (${last.yearParamsB?.toFixed?.(0) ?? '—'}B params) needs <strong>${fabricNeeded}</strong> coherent-fabric GPUs, exceeding the ${plan.gpu?.label || plan.headlineSku} domain of ${scaleUpDomain}. Pipeline parallelism or a different SKU required.</span>`);
  } else {
    lines.push(`<span class="good">✓ Frontier model fits in a ${fabricNeeded}-GPU scale-up domain on ${plan.gpu?.label || plan.headlineSku}.</span>`);
  }
  if (plan.anyShortfall) {
    lines.push(`<span class="warn">⚠ Site capacity insufficient in ≥1 year — consider cloud overflow or earlier site builds (2–3 yr lead times).</span>`);
  }

  const el = document.getElementById('narrative');
  el.innerHTML = lines.map(l => `<p>${l}</p>`).join('');
}

function initNarrativeToggle() {
  const sec = document.getElementById('narrative-section');
  document.getElementById('narrative-toggle').addEventListener('click', () =>
    sec.classList.toggle('collapsed'));
}

// ─── Insight pills ─────────────────────────────────────────────────────────
function initInsightPills() {
  document.querySelectorAll('.insight-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const key = pill.dataset.insight;
      const box = document.getElementById(`insight-${key}`);
      if (!box) return;
      const open = box.classList.toggle('open');
      pill.querySelector('.insight-pill-icon').textContent = open ? '✕' : '◈';
    });
  });
}

// ─── Recompute ───────────────────────────────────────────────────────────────
function recompute() {
  const t0 = performance.now();
  let plan;
  try { plan = buildPlan(currentInputs); }
  catch(e) { console.error('buildPlan error', e); return; }
  const elapsed = performance.now() - t0;

  setKpi('kpi-peak-gpus', fmt(plan.peakGpus));
  setKpi('kpi-peak-mw', plan.peakMw.toFixed(1) + ' MW');
  setKpi('kpi-peak-facilities', String(plan.peakFacilities));
  setKpi('kpi-tco', '$' + fmt(plan.totalDiscountedTco));
  const last = plan.yearly[plan.yearly.length - 1];
  setKpi('kpi-internal-price', '$' + last.internalDollarsPerMtok.toFixed(2) + '/Mtok');
  setKpi('kpi-fabric', last.minScaleUpGpus + ' GPUs');

  document.getElementById('compute-time').textContent = `${elapsed.toFixed(0)}ms`;

  renderDomainStack('chart-domain', plan);
  renderFp64Split('chart-fp64-split', plan);
  renderBoundHeatmap('chart-bound-heatmap', plan);
  renderSkuMix('chart-sku-mix', plan);
  renderFleetByWorkload('chart-fleet', plan);
  renderTcoStack('chart-tco', plan);
  renderBuildVsBuy('chart-buybuild', plan);
  renderFacilities('chart-facilities', plan);
  renderPower('chart-power', plan);
  renderScaleUp('chart-scaleup', plan);
  const frontier = latencyCostFrontier(currentInputs, 0);
  renderLatencyFrontier('chart-latency', frontier);

  updateNarrative(plan);

  setTimeout(() => {
    try {
      const sFan = scalingCaseFan(currentInputs);
      renderScalingCaseFan('chart-scaling-fan', sFan);
    } catch(e) { console.warn('scalingCaseFan error', e); }
  }, 50);
  setTimeout(() => {
    try {
      const fan = demandFan(currentInputs);
      renderDemandFan('chart-fan', fan);
      const tor = tornadoBattery(currentInputs, plan.totalDiscountedTco);
      renderTornado('chart-tornado', tor, plan.totalDiscountedTco);
    } catch(e) { console.warn('tornado/fan error', e); }
  }, 100);
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ─── Init ─────────────────────────────────────────────────────────────────────
export function init() {
  const tsEl = document.getElementById('gw-timestamp');
  if (tsEl) {
    const now = new Date();
    tsEl.textContent = now.toISOString().slice(0,16).replace('T',' ') + ' UTC';
  }

  buildGateway();
  buildHatPillsHeader();
  buildKpiRow();
  buildScenarioCards();
  buildPortfolioEditor();
  buildParamsControls();
  initSidebarTabs();
  initNarrativeToggle();
  initInsightPills();
  document.getElementById('recompute-btn').addEventListener('click', recompute);

  currentInputs = { ...SCENARIOS[DEFAULT_SCENARIO].inputs };
  applyInputsToDom(currentInputs);
}
