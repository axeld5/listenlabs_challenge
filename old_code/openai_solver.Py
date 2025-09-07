from __future__ import annotations

"""
Online quota-aware acceptance solver
-----------------------------------

Goal
====
Minimize rejections while accepting exactly N people (default 1000) arriving
sequentially with binary attributes, subject to lower-bound proportion
constraints of the form: for each attribute k, at least alpha[k] fraction of
accepted people must have x_k == 1.

We cannot see the future, but we know (or can estimate) the attribute
marginals P(x_k = 1) and (optionally) their pairwise correlations/covariances
for the incoming population. The solver operates online: at each arrival x,
it decides to accept or reject.

Core ideas
==========
1) **Hard-feasibility guard**: Never accept a candidate if, after accepting
them, it would be *impossible* to meet the final quotas even with perfect
future choices. This is a deterministic necessary condition and ensures we
never paint ourselves into a corner.

   For each attribute k, let t_k = ceil(alpha_k * N) be the final required
   count of ones. If we've currently accepted a people and have counts c_k,
   and there are r = N - a remaining acceptance slots, then accepting x
   (with x_k in {0,1}) is feasible only if for all k:

       c_k + x_k + (r - 1) >= t_k

   Equivalently, we must not accept zeros on attributes where the remaining
   required ones already exceed the remaining slots *after* this acceptance.

2) **Stochastic rollout (optional)**: To reduce unnecessary rejections,
   we can simulate the remainder of the process using the known marginals
   (and optional correlations) to estimate:
   - The probability we can still satisfy all quotas by N.
   - The *expected extra rejections* needed to reach N.

   We compare two branches — accept vs reject this candidate now — and choose
   the one with lower expected rejections subject to a maximum failure risk
   (chance constraints). This is a one-step lookahead policy that is fast and
   practical.

3) **Sampling arrivals**: If pairwise correlations are available we use a
   Gaussian copula approximation to generate correlated binary vectors with
   the desired marginals. If not, we sample attributes independently.

Usage
=====
- Construct the solver with your constraint vector `alphas`, marginal
  probabilities `p`, and optionally `cov` or `corr`.
- Call `decide(x)` for each new candidate vector (NumPy array of 0/1), then
  call `update(x, accept)` with the returned decision.
- Repeat until `accepted == N` or `rejected == reject_budget`.

See the `__main__` demo at the bottom for a runnable simulation example.

Note: This file has no external dependencies beyond NumPy.
"""

from dataclasses import dataclass
import math
import numpy as np
from typing import Optional, Tuple, Callable, Dict


# ----------------------------- Utilities ---------------------------------- #

def _ceil_int(x: float) -> int:
    return int(math.ceil(x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Fast inverse CDF for standard normal (Acklam's approximation).

    Works for scalar or array-like `p` in (0, 1). Values are clipped to
    [1e-12, 1-1e-12] for numerical stability.
    """
    # Coefficients in rational approximations
    a = np.array([
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ])
    b = np.array([
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ])
    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ])
    d = np.array([
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ])

    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    # Define break-points
    plow = 0.02425
    phigh = 1 - plow

    z = np.empty_like(p)

    # lower region
    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2 * np.log(p[mask]))
        z[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        z[mask] = -z[mask]

    # upper region
    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2 * np.log(1 - p[mask]))
        z[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    # central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q*q
        z[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    return z


def _nearest_psd(corr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semidefinite matrix
    by clipping negative eigenvalues, then renormalize diagonal to 1.
    """
    B = (corr + corr.T) / 2
    vals, vecs = np.linalg.eigh(B)
    vals = np.maximum(vals, eps)
    B_psd = (vecs * vals) @ vecs.T
    # Normalize to correlation (diag = 1)
    d = np.sqrt(np.clip(np.diag(B_psd), eps, None))
    B_corr = B_psd / d[:, None] / d[None, :]
    return (B_corr + B_corr.T) / 2


def build_correlation_from_cov(p: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Convert Bernoulli covariance to correlation matrix, then project to PSD.

    cov_ij = Corr_ij * sqrt(p_i(1-p_i) p_j(1-p_j))
    => Corr_ij = cov_ij / sqrt(var_i var_j)
    """
    p = np.asarray(p, dtype=float)
    var = p * (1 - p)
    denom = np.sqrt(np.maximum(var, 1e-12))
    corr = cov / (denom[:, None] * denom[None, :])
    # Clamp numerical range and set diag to 1
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -0.999, 0.999)
    return _nearest_psd(corr)


class ArrivalSampler:
    """Sampler for future arrivals.

    If `corr` is provided, use a Gaussian copula with the given correlation
    matrix to couple attributes; otherwise sample attributes independently.
    """
    def __init__(self, p: np.ndarray, corr: Optional[np.ndarray] = None, seed: Optional[int] = None):
        self.p = np.asarray(p, dtype=float)
        self.m = self.p.size
        self.rng = np.random.default_rng(seed)
        self.use_copula = corr is not None
        if self.use_copula:
            corr = np.asarray(corr, dtype=float)
            # Project to PSD & ensure correlation-like properties
            self.corr = _nearest_psd(corr)
            self.thresholds = _norm_ppf(self.p)
            # Cholesky may still fail if near-singular; fall back to eigh
            try:
                self.L = np.linalg.cholesky(self.corr)
                self._mvnorm = self._mvnorm_chol
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(self.corr)
                vals = np.clip(vals, 0.0, None)
                self.L_ev = vecs @ np.diag(np.sqrt(vals))
                self._mvnorm = self._mvnorm_eig
        else:
            self.corr = None

    def _mvnorm_chol(self, n: int = 1) -> np.ndarray:
        z = self.rng.standard_normal(size=(n, self.m))
        return z @ self.L.T

    def _mvnorm_eig(self, n: int = 1) -> np.ndarray:
        z = self.rng.standard_normal(size=(n, self.m))
        return z @ self.L_ev.T

    def sample(self, n: int = 1) -> np.ndarray:
        if not self.use_copula:
            # independent Bernoulli
            u = self.rng.random(size=(n, self.m))
            return (u < self.p).astype(np.int8)
        else:
            z = self._mvnorm(n)
            # Dichotomize at per-dimension thresholds to get Bernoulli
            return (z <= self.thresholds).astype(np.int8)


# --------------------------- Solver classes -------------------------------- #

@dataclass
class SolverConfig:
    alphas: np.ndarray            # shape (m,)
    p: np.ndarray                 # shape (m,), P(x_k=1)
    N: int = 1000                 # total acceptances target
    reject_budget: int = 20000    # max rejections allowed (stopping condition in sim)
    # If you pass `cov` we convert it to correlation; if you have `corr` already
    # you can pass it directly and leave cov=None.
    cov: Optional[np.ndarray] = None    # shape (m,m) Bernoulli covariance
    corr: Optional[np.ndarray] = None   # shape (m,m) Pearson correlation
    risk_tolerance: float = 0.01        # max failure prob in rollout (chance constraint)
    sims: int = 0                       # 0 => deterministic-only; >0 enables rollout lookahead
    seed: Optional[int] = None


class OnlineQuotaSolver:
    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self.m = int(cfg.alphas.size)
        self.alphas = np.asarray(cfg.alphas, dtype=float)
        assert np.all((self.alphas >= 0) & (self.alphas <= 1)), "alphas must be in [0,1]"
        self.p = np.asarray(cfg.p, dtype=float)
        assert self.p.size == self.m, "p and alphas must have same length"
        assert np.all((self.p > 0) & (self.p < 1)), "p must be strictly between 0 and 1 for all dims"
        if cfg.corr is not None:
            corr = np.asarray(cfg.corr, dtype=float)
        elif cfg.cov is not None:
            corr = build_correlation_from_cov(self.p, np.asarray(cfg.cov, dtype=float))
        else:
            corr = None
        self.sampler = ArrivalSampler(self.p, corr=corr, seed=cfg.seed)
        # Targets
        self.N = int(cfg.N)
        self.reject_budget = int(cfg.reject_budget)
        self.t = np.array([_ceil_int(a * self.N) for a in self.alphas], dtype=int)
        self.reset()

    # ----------------------------- State ---------------------------------- #
    def reset(self):
        self.a = 0                        # accepted so far
        self.r = 0                        # rejected so far
        self.c = np.zeros(self.m, dtype=int)  # counts of ones among accepted

    @property
    def remaining_slots(self) -> int:
        return self.N - self.a

    @property
    def remaining_required(self) -> np.ndarray:
        # Remaining required ones to hit targets, lower-bounded by 0
        return np.maximum(self.t - self.c, 0)

    # --------------------------- Core logic -------------------------------- #
    def _feasible_if_accept(self, x: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check the deterministic necessary feasibility after accepting x.
        Returns (feasible, per-attribute slack_after_accept), where slack is
        (c + x + r' - t). Negative slack for any k means infeasible.
        """
        x = np.asarray(x, dtype=int)
        r_after = self.remaining_slots - 1
        slack = self.c + x + r_after - self.t
        feasible = bool(np.all(slack >= 0))
        return feasible, slack

    def _greedy_safe_accept(self, x: np.ndarray) -> bool:
        feasible, _ = self._feasible_if_accept(x)
        return feasible

    def decide(self, x: np.ndarray) -> Tuple[bool, Dict]:
        """Return (accept_bool, info_dict) for the current candidate x.

        If cfg.sims == 0: purely deterministic — accept iff accepting is
        deterministically feasible. This is *very* fast and already performs
        well when you have a large rejection budget.

        If cfg.sims > 0: run a one-step stochastic rollout to choose the
        branch (accept vs reject) with lower expected rejections subject to a
        max failure probability (chance constraint).
        """
        x = np.asarray(x, dtype=int)
        assert x.size == self.m and np.all((x == 0) | (x == 1)), "x must be a 0/1 vector of length m"

        # Fast guard: if no slots left, must reject
        if self.remaining_slots <= 0:
            return False, {"reason": "no_slots_left"}

        # Deterministic necessary condition
        feasible, slack = self._feasible_if_accept(x)
        if not feasible and self.cfg.sims == 0:
            return False, {"reason": "infeasible_if_accept", "slack": slack}
        if feasible and self.cfg.sims == 0:
            return True, {"reason": "deterministic_feasible", "slack": slack}

        # If using rollout, evaluate both branches
        # Early pruning: if infeasible to accept, only 'reject' branch is valid
        branches = ["accept", "reject"] if feasible else ["reject"]
        best_choice = None
        best_metric = None
        details = {}

        for br in branches:
            ok, success_rate, exp_rej = self._rollout_metric(x, branch=br, sims=self.cfg.sims)
            details[br] = {"success_rate": success_rate, "exp_future_rejections": exp_rej}
            if not ok:
                continue
            metric = exp_rej  # we minimize expected future rejections
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_choice = br

        if best_choice is None:
            # Neither branch met risk tolerance; fall back to deterministic guard
            return (feasible, {"reason": "fallback_deterministic", "slack": slack, "rollout": details})

        return (best_choice == "accept", {"reason": "rollout_choice", "rollout": details})

    def update(self, x: np.ndarray, accept: bool):
        x = np.asarray(x, dtype=int)
        if accept:
            assert self._feasible_if_accept(x)[0], "update called with infeasible accept; guard with decide() first"
            self.a += 1
            self.c += x
        else:
            self.r += 1

    # ---------------------------- Rollout ---------------------------------- #
    def _clone_state(self):
        return self.a, self.r, self.c.copy()

    def _restore_state(self, snapshot):
        self.a, self.r, self.c = snapshot[0], snapshot[1], snapshot[2]

    def _rollout_metric(self, x: np.ndarray, branch: str, sims: int) -> Tuple[bool, float, float]:
        """Simulate the remainder under the greedy-safe policy starting with
        either accepting or rejecting the current x. Returns:
            (ok, success_rate, expected_future_rejections)
        where `ok` is True if success_rate >= 1 - risk_tolerance.
        """
        assert branch in ("accept", "reject")
        successes = 0
        rejections_list = []
        snapshot = self._clone_state()

        for _ in range(sims):
            self._restore_state(snapshot)
            # Apply the branch decision
            if branch == "accept":
                if not self._feasible_if_accept(x)[0]:
                    # Immediate failure for this path
                    success = False
                    future_rej = 0
                    rejections_list.append(future_rej)
                    continue
                self.a += 1
                self.c += x
            else:
                self.r += 1

            # Simulate until we hit N or the reject budget (only matters if you
            # want to guard against exhausting allowed rejections)
            while self.a < self.N and self.r < self.reject_budget:
                y = self.sampler.sample(1)[0]
                if self._greedy_safe_accept(y):
                    self.a += 1
                    self.c += y
                else:
                    self.r += 1

            success = (self.a >= self.N) and np.all(self.c >= self.t)
            future_rej = max(0, self.r - snapshot[1] - (1 if branch == "reject" else 0))
            rejections_list.append(future_rej)
            if success:
                successes += 1

        self._restore_state(snapshot)
        success_rate = successes / sims if sims > 0 else 1.0
        exp_rej = float(np.mean(rejections_list)) if sims > 0 else 0.0
        ok = (success_rate >= 1 - self.cfg.risk_tolerance)
        return ok, success_rate, exp_rej


# ------------------------------- Demo -------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo runner for OnlineQuotaSolver")
    parser.add_argument("--m", type=int, default=6, help="Number of attributes")
    parser.add_argument("--N", type=int, default=1000, help="Target accepts")
    parser.add_argument("--sims", type=int, default=0, help="Rollout simulations per decision (0 => deterministic only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--T", type=int, default=50000, help="Max arrivals to simulate in demo")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    m = args.m

    # Synthetic problem
    p = rng.uniform(0.2, 0.8, size=m)
    alphas = np.clip(p + rng.normal(0, 0.05, size=m), 0.05, 0.95) * 0.7  # make quotas achievable

    # Optional weak positive correlations
    base = rng.normal(size=(m, m))
    Sigma = base @ base.T
    D = np.diag(1 / np.sqrt(np.diag(Sigma)))
    corr = D @ Sigma @ D
    corr = 0.2 * corr + 0.8 * np.eye(m)  # shrink toward identity

    cfg = SolverConfig(alphas=alphas, p=p, N=args.N, sims=args.sims, corr=corr, seed=args.seed)
    solver = OnlineQuotaSolver(cfg)

    accepted, rejected = 0, 0
    arrivals = 0
    while solver.a < solver.N and arrivals < args.T:
        x = solver.sampler.sample(1)[0]
        accept, info = solver.decide(x)
        solver.update(x, accept)
        arrivals += 1
        if arrivals % 1000 == 0:
            print(f"Arrivals={arrivals}, accepted={solver.a}, rejected={solver.r}, needed={solver.remaining_required.sum()} total")

    print("\n=== Summary ===")
    print(f"Arrivals seen: {arrivals}")
    print(f"Accepted: {solver.a} / {solver.N}")
    print(f"Rejected: {solver.r}")
    print(f"Final counts per attr: {solver.c}")
    print(f"Targets per attr:     {solver.t}")
    print(f"Feasible: {np.all(solver.c >= solver.t)}")
