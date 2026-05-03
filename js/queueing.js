// Queueing models for capacity sizing under p99 latency targets.
//
// Average utilization sets average latency. P99 latency under load is what
// users actually care about, and the gap between average and p99 grows
// nonlinearly with utilization. We provide:
//   - a simple sqrt-c heuristic for the maximum sustainable utilization
//     (captures statistical multiplexing as fleet size grows)
//   - a Sakasegawa-style M/M/c approximation for the implied p99 wait

// Heuristic utilization ceiling that respects p99 latency.
// Small fleets need lots of headroom; big fleets approach high utilization
// thanks to statistical multiplexing across many concurrent requests.
//   c=1     -> ~0.40
//   c=10    -> ~0.68
//   c=100   -> ~0.90
//   c=1000  -> ~0.97
export function utilizationCeiling({
  cServers = 1,
  floor = 0.4,
  ceiling = 0.95,
} = {}) {
  if (cServers <= 0) return floor;
  const u = 1 - 1 / Math.sqrt(cServers);
  return Math.max(floor, Math.min(ceiling, u));
}

// Sakasegawa approximation for p99 wait time in an M/M/c queue.
// Useful for showing whether a chosen utilization meets a latency budget.
//   E[Wq] ≈ ρ^(sqrt(2(c+1))) / (c*(1-ρ)) * E[S]
// P99(Wq) ≈ -ln(0.01) * E[Wq]   (exponential-tail approximation)
export function p99WaitMs({ rho, cServers, serviceTimeMs }) {
  if (rho >= 1 || rho <= 0 || cServers <= 0) return Infinity;
  const exponent = Math.sqrt(2 * (cServers + 1));
  const eWq = (Math.pow(rho, exponent) / (cServers * (1 - rho))) * serviceTimeMs;
  return -Math.log(0.01) * eWq;
}

// Inverts the above approximately: given target p99 wait and service time,
// find the max ρ that satisfies it for the given c. Bisection.
export function maxUtilForP99({ cServers, serviceTimeMs, targetP99WaitMs }) {
  if (targetP99WaitMs >= Infinity) return 0.95;
  let lo = 0.01, hi = 0.99;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    const w = p99WaitMs({ rho: mid, cServers, serviceTimeMs });
    if (w > targetP99WaitMs) hi = mid;
    else lo = mid;
  }
  return Math.max(0.4, Math.min(0.95, lo));
}
