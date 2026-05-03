// Portfolio: a mix of workload-class instances the buyer wants to compose.
//
// A portfolio item is a thin wrapper around a workload class that records:
//   - which class it is (id pointing into WORKLOAD_CLASSES)
//   - the buyer's demand specification (unitsPerYear)
//   - which scaling case to use ("low" | "median" | "high" | "fitted")
//   - any class-specific overrides (e.g. flops_per_unit if the buyer has
//     a more accurate measurement for their model)
//   - whether this item is enabled
//
// Growth models translate (year, scalingCase) into a multiplier on demand.

import { WORKLOAD_CLASSES } from "./workload_classes.js";

export function makePortfolioItem(classId, opts = {}) {
  const cls = WORKLOAD_CLASSES[classId];
  if (!cls) throw new Error(`Unknown workload class: ${classId}`);
  return {
    classId,
    enabled: opts.enabled !== false,
    unitsPerYear: opts.unitsPerYear ?? cls.default_demand?.unitsPerYear ?? 0,
    scalingCase: opts.scalingCase ?? "median",
    growthModel: opts.growthModel ?? cls.growth_model_default ?? "exponential",
    overrides: opts.overrides ?? {},          // e.g., { flops_per_unit: 5e15 }
  };
}

// Resolve a portfolio item into the live workload object the dispatcher wants.
export function resolveWorkload(item) {
  const cls = WORKLOAD_CLASSES[item.classId];
  if (!cls) return null;
  return { ...cls, ...item.overrides };
}

// Per-item demand at year y, given growth model and scaling case.
export function demandAtYear(item, y) {
  const cls = WORKLOAD_CLASSES[item.classId];
  if (!cls || !item.enabled) return 0;
  const base = item.unitsPerYear;
  if (base <= 0) return 0;

  switch (item.growthModel) {
    case "exponential": {
      const rate = cls.scaling_uncertainty?.[item.scalingCase]?.yoy ?? 0.3;
      return base * Math.pow(1 + rate, y);
    }
    case "scaling_law": {
      // For LLM-class items, "fitted" uses Hoffmann compute-doubling-every-6mo;
      // for other classes, falls through to the case-specific yoy rate.
      if (item.scalingCase === "fitted" && cls.domain === "llm") {
        // Compute doubles every 6 months -> 4× per year FLOPs growth; demand
        // tokens grow more slowly (~2× / year) because algorithmic efficiency
        // amortizes the rest. We use 1.0 (i.e., 2x/yr) as the fitted default.
        return base * Math.pow(2.0, y);
      }
      const rate = cls.scaling_uncertainty?.[item.scalingCase]?.yoy ?? 0.5;
      return base * Math.pow(1 + rate, y);
    }
    case "linear": {
      const addPct = cls.growth_params?.yoyAddPercent ?? 0.10;
      return base * (1 + addPct * y);
    }
    case "campaign": {
      // Campaign-shaped growth: average annual demand × (1 + yoy) growth.
      const yoy = cls.growth_params?.growthYoy
                 ?? cls.scaling_uncertainty?.[item.scalingCase]?.yoy
                 ?? 0.3;
      return base * Math.pow(1 + yoy, y);
    }
    case "steady": {
      return base;
    }
    case "latent": {
      // Latent / unfitted scaling: use the explicit case-yoy from the class.
      const rate = cls.scaling_uncertainty?.[item.scalingCase]?.yoy ?? 0.3;
      return base * Math.pow(1 + rate, y);
    }
    default:
      return base;
  }
}

// Convenience: legacy aggregator -> produce { interactiveTokensPerDay,
// batchTokensPerDay, rlTokensPerDay, pretrainFlopsPerYear, finetuneFlopsPerYear }
// from a portfolio. Used so the existing buildPlan path can keep working
// during the transition; long-term, model.js should consume the portfolio
// directly.
export function legacyDemandFromPortfolio(portfolio, year = 0) {
  let interactive = 0, batch = 0, rl = 0, pretrain = 0, finetune = 0;
  for (const item of portfolio) {
    const cls = WORKLOAD_CLASSES[item.classId];
    if (!cls || !item.enabled) continue;
    const demand = demandAtYear(item, year);
    switch (cls.legacy_category) {
      case "interactive_inference": interactive += demand; break;
      case "batch_inference":       batch       += demand; break;
      case "rl":                    rl          += demand; break;
      case "training":              pretrain    += demand; break;
      case "finetune":              finetune    += demand; break;
    }
  }
  // Convert tokens/year to tokens/day for the legacy fields.
  return {
    interactiveTokensPerDay: interactive / 365,
    batchTokensPerDay: batch / 365,
    rlTokensPerDay: rl / 365,
    pretrainFlopsPerYear: pretrain,
    finetuneFlopsPerYear: finetune,
  };
}

// A blank starter portfolio: just two LLM workloads enabled.
export const STARTER_PORTFOLIO = [
  makePortfolioItem("llm_interactive", { unitsPerYear: 5e9 * 365 }),
  makePortfolioItem("llm_batch",       { unitsPerYear: 20e9 * 365 }),
];
