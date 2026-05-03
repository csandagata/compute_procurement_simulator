// Regional capacity supply curves.
//
// In practice, datacenter MW is the binding constraint for hyperscalers.
// New facilities take 24-36 months from breaking ground to first GPUs powered.
// Each region has its own price stack and lead time.

export const DEFAULT_REGIONS = [
  {
    id: "us-central",
    label: "US Central (cheap power, hyperscale)",
    pue: 1.18,
    electricityPrice: 0.045,
    mwAvailableByYear: [80, 200, 400, 700, 1000, 1400, 1800],
    leadTimeYears: 2,
    networkTierFloor: "large_pod",
  },
  {
    id: "us-west",
    label: "US West / Pacific Northwest (hydro)",
    pue: 1.20,
    electricityPrice: 0.055,
    mwAvailableByYear: [40, 80, 150, 250, 400, 600, 800],
    leadTimeYears: 2,
    networkTierFloor: "large_pod",
  },
  {
    id: "us-east",
    label: "US East (low-latency edge)",
    pue: 1.30,
    electricityPrice: 0.10,
    mwAvailableByYear: [30, 50, 80, 120, 180, 250, 320],
    leadTimeYears: 3,
    networkTierFloor: "small_pod",
  },
  {
    id: "europe",
    label: "Europe (sovereignty, regulated)",
    pue: 1.30,
    electricityPrice: 0.16,
    mwAvailableByYear: [20, 35, 60, 100, 140, 200, 260],
    leadTimeYears: 3,
    networkTierFloor: "small_pod",
  },
  {
    id: "neocloud",
    label: "Neocloud rental (instant overflow)",
    pue: 1.35,
    electricityPrice: 0.13,
    mwAvailableByYear: [200, 500, 1000, 2000, 3500, 5000, 7000],
    leadTimeYears: 0,
    networkTierFloor: "small_pod",
    isRental: true,
  },
];

// Place demand on regions according to a policy.
// policy.tierShares maps a latency tier (edge/regional/central) to the fraction of
// demand that must land in that tier; we then route within the tier preferring
// the cheapest still-available region.
//
// Returns per-year placement: which regions hold how many MW, plus shortfall.
export function placeMwAcrossRegions({
  mwNeededByYear,
  regions = DEFAULT_REGIONS,
  preferredOrder,        // optional: array of region ids in preference order
  rentalShareOverride,   // optional: force a fraction onto neocloud, regardless of price
} = {}) {
  const order = preferredOrder
    || regions.slice().sort((a, b) => a.electricityPrice - b.electricityPrice).map((r) => r.id);

  const out = [];
  for (let y = 0; y < mwNeededByYear.length; y++) {
    const totalNeed = mwNeededByYear[y];
    const placements = regions.map((r) => ({
      regionId: r.id, label: r.label, mwUsed: 0,
      mwCapacity: r.mwAvailableByYear[Math.min(y, r.mwAvailableByYear.length - 1)] || 0,
      isRental: !!r.isRental, atCapacity: false,
      electricityPrice: r.electricityPrice, pue: r.pue, leadTimeYears: r.leadTimeYears,
    }));
    const byId = Object.fromEntries(placements.map((p) => [p.regionId, p]));

    let remaining = totalNeed;

    // 1) honor rental override first
    if (rentalShareOverride && rentalShareOverride > 0) {
      const target = totalNeed * rentalShareOverride;
      const rentalPlacements = placements.filter((p) => p.isRental);
      let toPlace = target;
      for (const p of rentalPlacements) {
        const used = Math.min(p.mwCapacity, toPlace);
        p.mwUsed = used; toPlace -= used; remaining -= used;
        if (used >= p.mwCapacity) p.atCapacity = true;
      }
    }

    // 2) place the rest in preferred order, skipping at-capacity regions
    for (const id of order) {
      if (remaining <= 0) break;
      const p = byId[id];
      if (!p || p.isRental) continue; // rentals only via override
      const free = Math.max(0, p.mwCapacity - p.mwUsed);
      const used = Math.min(free, remaining);
      p.mwUsed += used;
      remaining -= used;
      if (p.mwUsed >= p.mwCapacity) p.atCapacity = true;
    }

    // 3) overflow to neocloud (involuntary rental)
    if (remaining > 0) {
      const rentals = placements.filter((p) => p.isRental);
      for (const p of rentals) {
        const free = Math.max(0, p.mwCapacity - p.mwUsed);
        const used = Math.min(free, remaining);
        p.mwUsed += used;
        remaining -= used;
        if (p.mwUsed >= p.mwCapacity) p.atCapacity = true;
      }
    }

    out.push({
      year: y,
      mwNeeded: totalNeed,
      regions: placements,
      shortfall: Math.max(0, remaining),
    });
  }
  return out;
}
