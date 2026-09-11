"""Numerical verification of compute_adaptive_LACC.optimize_one_position.

Cross-checks the active-set enumeration candidates against an independent
brute-force search of the exact objective, for random instances matching the
production shape (N members, T lag times, sigma).

An INDEPENDENT c, S is computed here from the docstring formula, so any bug in
the code's covariance construction would show up as a mismatch between the
code's selected score and the reference optimum.
"""
import sys
import types

# ---- stub netCDF4 so the module imports without the dependency
_nc = types.ModuleType("netCDF4")
_nc.Dataset = object
sys.modules.setdefault("netCDF4", _nc)

sys.path.insert(0, "3create_obs/hx_rttov")
import compute_adaptive_LACC as cal
from compute_adaptive_LACC import (
    Config, optimize_one_position, _objective, _stationary_candidates,
)
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20260911)

SIGMA2 = 0.5 ** 2


# ---------- independent reference implementation ----------
def ref_cS(Y, s):
    """(N,T) -> c (T,), S (T,T) per the docstring formula."""
    N = Y.shape[0]
    s_p = s - s.mean()
    Y_p = Y - Y.mean(axis=0)
    c = Y_p.T @ s_p / (N - 1)
    S = Y_p.T @ Y_p / (N - 1) + SIGMA2 * np.eye(Y.shape[1])
    S = 0.5 * (S + S.T)
    return c, S


def ref_J(c, S, w):
    num = float(c @ w) ** 2
    den = float(w @ S @ w)
    return num / den if den > 0 else -np.inf


def simplex_grid(n, step):
    n_steps = int(round(1.0 / step))
    out = []

    def rec(prefix, rem):
        if len(prefix) == n - 1:
            out.append(prefix + [rem])
            return
        for v in range(rem + 1):
            rec(prefix + [v], rem - v)

    rec([], n_steps)
    g = np.asarray(out, float) * step
    g[:, -1] = np.clip(1.0 - g[:, :-1].sum(1), 0.0, 1.0)
    return g


def polish(c, S, w0):
    """Local exact-optimum search from a start point (SCS/SLSQP)."""
    w0 = np.asarray(w0, float)
    w0 = np.clip(w0, 0, None)
    w0 = w0 / w0.sum()
    res = minimize(
        lambda w: -ref_J(c, S, w), w0, method="SLSQP",
        bounds=[(0, 1)] * len(w0),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"maxiter": 3000, "ftol": 1e-14},
    )
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return w, ref_J(c, S, w)


def brute_best(c, S, n_times, n_random, dense_step):
    """Return (best value, best w) over dense grid + many random starts + polishing."""
    best, bestw = -np.inf, None
    if dense_step is not None:
        G = simplex_grid(n_times, dense_step)
        js = np.array([ref_J(c, S, w) for w in G])
        i = int(np.argmax(js))
        if js[i] > best:
            best, bestw = float(js[i]), G[i]
    W = rng.dirichlet(np.ones(n_times), size=n_random)
    js = np.array([ref_J(c, S, w) for w in W])
    i = int(np.argmax(js))
    if js[i] > best:
        best, bestw = float(js[i]), W[i]
    # polish a few best starts (the top random, the grid best if any, the code candidate)
    starts = [bestw] + list(W[np.argsort(js)[-(max(1, min(20, n_random // 1000))):]])
    for st in starts:
        w, v = polish(c, S, st)
        if v > best:
            best, bestw = float(v), w
    return best, bestw


def make_instance(T, N=50):
    """Realistic-magnitude Y (BT in K) and s (SST in K) with signal structure."""
    members = rng.normal(0, 1.5, size=(N, T))          # member per-lag perturbations
    base = rng.uniform(200, 260, size=T)                 # lag-mean BT
    Y = base[None, :] + members
    w0 = rng.dirichlet(np.ones(T))                       # hidden "true" time weights
    noise_scale = 10.0 ** rng.uniform(-2.0, -0.3)        # signal-to-noise varied
    s = Y @ w0 + rng.normal(0, noise_scale, size=N)      # target correlated with Y
    # occasionally make target nearly unrelated
    if rng.random() < 0.15:
        s = rng.normal(0, 1.0, size=N)
    return Y, s


def run():
    cfg = Config(obs_err_std=0.5)
    worst_gap = 0.0
    worst_gap_info = None
    n_trials = 200
    score_mismatch = []   # code-reported score vs independent J at same w
    fails = []
    TOL = 1e-8
    for T in (2, 3, 4, 5, 6):
        dense = {2: 0.002, 3: 0.005, 4: 0.008}.get(T, None)
        n_rand = {2: 80000, 3: 80000, 4: 80000, 5: 300000, 6: 400000}[T]
        for trial in range(n_trials):
            Y, s = make_instance(T)
            c, S = ref_cS(Y, s)
            w_code, sc_code, sc_eq, fb, reason, maxcand, solver = optimize_one_position(
                Y, s, SIGMA2, cfg
            )
            # simplify: only test cases the code actually runs the enumerator on
            if solver == "equal_fallback" or solver == "t1_direct":
                continue
            # feasibility
            assert np.all(w_code >= -1e-9), (T, trial, w_code)
            assert abs(w_code.sum() - 1.0) < 1e-8, (T, trial, w_code.sum())
            # code's own reported score must equal independent J at w_code
            j_ind = ref_J(c, S, w_code)
            assert j_ind > -1e300
            if abs(sc_code - j_ind) > TOL:
                score_mismatch.append((T, trial, sc_code - j_ind))
                fails.append((T, trial, "code-score-vs-independent", sc_code, j_ind))
            # brute-force reference
            best, bestw = brute_best(c, S, T, n_rand, dense)
            gap = best - sc_code
            worst_gap = max(worst_gap, gap)
            if gap > TOL:
                worst_gap_info = (T, trial, gap, sc_code, best)
                fails.append((T, trial, "gap>0", sc_code, best))
            if len(fails) > 12:
                break
        print(f"T={T}: checked {n_trials} instances; fails so far: {len(fails)}")
    print(f"\nRESULT worst_gap={worst_gap:.3e} (TOL={TOL})")
    print(f"worst_gap_info={worst_gap_info}")
    n_score_mismatch = len(score_mismatch)
    print(f"score_mismatch_count={n_score_mismatch}")
    if n_score_mismatch:
        print(f"score_mismatch sample: {score_mismatch[:5]}")
    print(f"FAILS: {fails[:12]}")
    return len(fails) == 0


if __name__ == "__main__":
    ok = run()
    print("\nPASSED" if ok else "\nFAILED")