// Speculative decoding speedup model.
//
// A draft model proposes gamma+1 tokens; the target model verifies them in one
// forward pass. Each speculative token is accepted with probability ~p (i.i.d.
// approximation following Leviathan et al. 2023).
//
// Expected accepted tokens per verification step:
//   E[A] = sum_{k=0..gamma} p^k = (1 - p^(gamma+1)) / (1 - p)
//
// The verifier processes (gamma+1) tokens together; in the memory-bound decode
// regime that's nearly free in time. The cost is one draft pass per speculative
// token, modeled as a fraction of the target forward-pass cost.

export function specDecSpeedup({
  acceptanceProb = 0,
  gammaMax = 0,
  draftToTargetCostRatio = 0.08,    // ~ 8B draft vs 70B target -> ~10%
} = {}) {
  if (gammaMax <= 0 || acceptanceProb <= 0) return 1.0;
  const p = Math.max(0, Math.min(0.999, acceptanceProb));
  const gamma = Math.max(1, Math.floor(gammaMax));
  const expectedAccepted = (1 - Math.pow(p, gamma + 1)) / (1 - p);
  // Verifier cost is ~1× target step (memory-bound); draft cost adds gamma × ratio.
  const stepCost = 1 + gamma * draftToTargetCostRatio;
  return expectedAccepted / stepCost;
}

// Returns the speedup along with bookkeeping for diagnostics.
export function specDecDetails(opts) {
  const speedup = specDecSpeedup(opts);
  const p = opts.acceptanceProb || 0;
  const gamma = opts.gammaMax || 0;
  const expectedAccepted = (gamma > 0 && p > 0)
    ? (1 - Math.pow(p, gamma + 1)) / (1 - p)
    : 1;
  return { speedup, expectedAccepted, p, gamma };
}
