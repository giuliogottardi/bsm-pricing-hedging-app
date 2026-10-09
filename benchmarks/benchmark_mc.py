"""Small reproducible benchmark for vectorized Monte Carlo vs per-path Python loop.
Run from repository root: python benchmarks/benchmark_mc.py
This is a local performance diagnostic, not a stable cross-machine benchmark.
"""
from time import perf_counter
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import core

PARAMS = dict(S0=100.0, K=100.0, T=0.5, r=0.03, vol=0.25,
              option_type="call", step=1e-4, rebalancing_frequency="weekly",
              custom_rebalancing_days=3)
N_SIMS = 500

start = perf_counter()
vectorized, _ = core.run_mc_batch(**PARAMS, n_sims=N_SIMS, seed=2024, keep_paths=0)
vectorized_seconds = perf_counter() - start

start = perf_counter()
rng = np.random.default_rng(2024)
looped = np.empty(N_SIMS)
for i in range(N_SIMS):
    result = core.run_delta_hedge_simulation(**PARAMS, rng=rng)
    looped[i] = result["hedging_pnl"]
looped_seconds = perf_counter() - start

print(f"Paths: {N_SIMS}; steps/path: {max(round(PARAMS['T'] * 252), 2)}")
print(f"Vectorized batch: {vectorized_seconds:.3f}s")
print(f"Per-path loop:    {looped_seconds:.3f}s")
print(f"Observed speed ratio: {looped_seconds / vectorized_seconds:.2f}x")
print(f"Maximum absolute pathwise P&L difference: {np.max(np.abs(vectorized - looped)):.3e}")
print("Distribution diagnostics:")
print(f"  vectorized mean/std = {np.mean(vectorized):.5f} / {np.std(vectorized, ddof=1):.5f}")
print(f"  looped mean/std     = {np.mean(looped):.5f} / {np.std(looped, ddof=1):.5f}")
