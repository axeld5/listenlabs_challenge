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
    # --- Performance knobs ---
    rollout_gate_slack: int = 8         # only run rollout when min slack < this (tight region)
    rollout_every: int = 1              # run rollout every K-th decision (use >1 to thin)
    rollout_batch: int = 512            # sample this many arrivals per RNG call in rollout


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
        self._decisions = 0               # counter for gating rollout frequency

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
        deterministically feasible.

        If cfg.sims > 0: optionally run a one-step stochastic rollout to choose
        the branch (accept vs reject) with lower expected rejections, subject to
        a max failure probability.
        """
        x = np.asarray(x, dtype=int)
        assert x.size == self.m and np.all((x == 0) | (x == 1)), "x must be a 0/1 vector of length m"
        self._decisions += 1

        # Fast guard: if no slots left, must reject
        if self.remaining_slots <= 0:
            return False, {"reason": "no_slots_left"}

        # Deterministic necessary condition for accepting x
        feasible_accept, slack_accept = self._feasible_if_accept(x)
        if self.cfg.sims == 0:
            return (feasible_accept, {"reason": "deterministic_only", "slack": slack_accept})

        # Compute min slack for the reject branch (keeping state unchanged)
        r_after = self.remaining_slots  # if we reject now
        slack_reject = self.c + r_after - self.t

        # Gating: only run rollout if we're in a "tight" region or at cadence
        tight = (feasible_accept and slack_accept.min() < self.cfg.rollout_gate_slack) or \
                (slack_reject.min() < self.cfg.rollout_gate_slack)
        cadence = (self._decisions % self.cfg.rollout_every == 0)

        if not tight or not cadence:
            # Cheap decision: accept if feasible, else reject
            return (feasible_accept, {"reason": "gated_no_rollout", "slack_accept": slack_accept, "slack_reject": slack_reject})

        # Rollout both branches (pruned if accept infeasible)
        branches = ["accept", "reject"] if feasible_accept else ["reject"]
        best_choice = None
        best_metric = None
        details = {}

        for br in branches:
            ok, success_rate, exp_rej = self._rollout_metric(x, branch=br, sims=self.cfg.sims)
            details[br] = {"success_rate": success_rate, "exp_future_rejections": exp_rej}
            if not ok:
                continue
            metric = exp_rej
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_choice = br

        if best_choice is None:
            # Neither branch met risk tolerance; fall back to deterministic guard
            return (feasible_accept, {"reason": "fallback_deterministic", "slack": slack_accept, "rollout": details})

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
        # IMPORTANT: copy the array to avoid aliasing across rollouts.
        # Otherwise, simulations would mutate the snapshot and leak state.
        self.a = snapshot[0]
        self.r = snapshot[1]
        self.c = snapshot[2].copy()

    def _rollout_metric(self, x: np.ndarray, branch: str, sims: int) -> Tuple[bool, float, float]:
        """Simulate the remainder under the greedy-safe policy starting with
        either accepting or rejecting the current x. Returns:
            (ok, success_rate, expected_future_rejections)
        where `ok` is True if success_rate >= 1 - risk_tolerance.
        Optimizations:
          - Early exit when all quotas are secured (no more rejections can happen).
          - Batch RNG: sample up to `rollout_batch` arrivals per call to the sampler.
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
                    rejections_list.append(0)
                    continue
                self.a += 1
                self.c += x
            else:
                self.r += 1

            # Simulate with early termination and batched sampling
            while self.a < self.N and self.r < self.reject_budget:
                # If all lower-bounds are already secured, we can accept everyone
                if np.all(self.c >= self.t):
                    # No further rejections will occur under greedy-safe policy
                    self.a = self.N
                    break

                batch_size = min(self.cfg.rollout_batch, 4 * (self.N - self.a))
                Y = self.sampler.sample(batch_size)
                for y in Y:
                    if self.a >= self.N or self.r >= self.reject_budget:
                        break
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

def build_inputs_from_user_data(population_data: dict, constraints_data: list, N: int = 1000):
    """Convert the user's dicts into (alphas, p, corr, attr_names).

    - population_data['relativeFrequencies'] maps attr -> P(x=1)
    - population_data['correlations'] is a nested dict corr[attr_i][attr_j]
    - constraints_data is a list of {"attribute": name, "minCount": int}
    """
    # Attribute order: follow the constraints order so minCounts align
    attr_names = [c['attribute'] for c in constraints_data]

    # Build p vector in that order
    rf = population_data['relativeFrequencies']
    p = np.array([rf[a] for a in attr_names], dtype=float)

    # Build correlation matrix in that order from nested dict
    corr_dict = population_data.get('correlations')
    if corr_dict is None:
        corr = None
    else:
        m = len(attr_names)
        corr = np.zeros((m, m), dtype=float)
        for i, ai in enumerate(attr_names):
            for j, aj in enumerate(attr_names):
                corr[i, j] = float(corr_dict[ai][aj])
        # Project to PSD just in case of small numerical issues
        corr = _nearest_psd(corr)

    # Alphas (quota proportions)
    t = np.array([int(c['minCount']) for c in constraints_data], dtype=int)
    alphas = t / float(N)
    return alphas, p, corr, attr_names


def run_user_simulation(population_data: dict, constraints_data: list, *, N: int = 1000,
                        reject_budget: int = 20000, sims: int = 400,
                        risk_tolerance: float = 0.02, seed: int = 123,
                        max_arrivals: int = 200000,
                        rollout_gate_slack: int = 8,
                        rollout_every: int = 1,
                        rollout_batch: int = 512):
    """Run a single streaming simulation with the provided data.

    - If sims == 0 the policy is deterministic (fastest).
    - If sims > 0 we use stochastic one-step rollout per candidate.
    """
    alphas, p, corr, names = build_inputs_from_user_data(population_data, constraints_data, N)

    cfg = SolverConfig(
        alphas=alphas,
        p=p,
        N=N,
        reject_budget=reject_budget,
        corr=corr,
        sims=sims,
        risk_tolerance=risk_tolerance,
        seed=seed,
        rollout_gate_slack=rollout_gate_slack,
        rollout_every=rollout_every,
        rollout_batch=rollout_batch,
    )
    solver = OnlineQuotaSolver(cfg)

    arrivals = 0
    while solver.a < solver.N and arrivals < max_arrivals and solver.r < solver.reject_budget:
        x = solver.sampler.sample(1)[0]
        accept, _ = solver.decide(x)
        solver.update(x, accept)
        arrivals += 1

    # Pretty print results
    print("=== Simulation Summary ===")
    print(f"Arrivals seen:        {arrivals}")
    print(f"Accepted:             {solver.a} / {solver.N}")
    print(f"Rejected:             {solver.r}")
    print("Per-attribute results:")
    for i, name in enumerate(names):
        print(f"  - {name:>15s}: count={solver.c[i]:4d}  target={solver.t[i]:4d}  alpha={alphas[i]:.3f}  p={p[i]:.4f}")
    feasible = bool(np.all(solver.c >= solver.t))
    print(f"Feasible (all quotas met): {feasible}")
    if not feasible:
        deficits = solver.t - solver.c
        for i, name in enumerate(names):
            if deficits[i] > 0:
                print(f"  deficit {name}: need {deficits[i]} more")


if __name__ == "__main__":
    # User-supplied data
    population_data = {
        'relativeFrequencies': {
            'techno_lover': 0.6265000000000001,
            'well_connected': 0.4700000000000001,
            'creative': 0.06227,
            'berlin_local': 0.398
        },
        'correlations': {
            'techno_lover': {'techno_lover': 1, 'well_connected': -0.4696169332674324, 'creative': 0.09463317039891586, 'berlin_local': -0.6549403815606182},
            'well_connected': {'techno_lover': -0.4696169332674324, 'well_connected': 1, 'creative': 0.14197259140471485, 'berlin_local': 0.5724067808436452},
            'creative': {'techno_lover': 0.09463317039891586, 'well_connected': 0.14197259140471485, 'creative': 1, 'berlin_local': 0.14446459505650772},
            'berlin_local': {'techno_lover': -0.6549403815606182, 'well_connected': 0.5724067808436452, 'creative': 0.14446459505650772, 'berlin_local': 1}
        }
    }

    constraints_data = [
        {'attribute': 'techno_lover', 'minCount': 650},
        {'attribute': 'well_connected', 'minCount': 450},
        {'attribute': 'creative', 'minCount': 300},
        {'attribute': 'berlin_local', 'minCount': 750}
    ]

    # Choose sims=0 for the ultra-fast deterministic policy,
    # or sims>0 for the probabilistic rollout to reduce expected rejections.
    run_user_simulation(population_data, constraints_data,
                        N=1000,
                        reject_budget=20000,
                        sims=400,              # try 0, 200, 400, 800 depending on speed/quality tradeoff
                        risk_tolerance=0.02,
                        seed=123,
                        max_arrivals=200000,
                        rollout_gate_slack=6,
                        rollout_every=10,
                        rollout_batch=2048)
