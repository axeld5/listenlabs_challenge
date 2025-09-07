"""
Berghain Challenge — Multi‑turn RL Environment (custom scenarios)

You are the bouncer at a nightclub. People arrive one by one with binary attributes.
Decide accept(1)/reject(0) immediately. Goal: fill the venue with N people while
meeting all minimum proportion/absolute constraints. Minimize rejections.

API: Gymnasium‑style environment.

- Observation: Dict with
    - 'attrs': MultiBinary(K) — current visitor's attributes (0/1)
    - 'accepted': Discrete(N+1)
    - 'rejected': Discrete(max_rejects+1)
- Action: Discrete(2) — 0=reject, 1=accept
- Termination:
    - Success: accepted == N and constraints satisfied  -> reward = -rejected_count
    - Failure: accepted == N and constraints NOT satisfied OR rejected == max_rejects -> reward = -100000

Three built‑in baseline scenarios remain, plus three new custom scenarios from the prompt.
Attributes are sampled i.i.d. with specified marginals and correlations via a Gaussian copula.

Usage
-----
from berghain_env_custom import BerghainEnv, make_env, SCENARIOS

env = make_env('custom_scenario_1', seed=42)
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(reward, info)

See bottom for example baseline policies and quick self‑test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # fallback to classic gym if needed
    import gym
    from gym import spaces


# ----------------------------
# Scenario configuration
# ----------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    attrs: List[str]  # attribute names in fixed order
    marginals: np.ndarray  # shape (K,), P(attr=1)
    corr: np.ndarray  # shape (K, K), correlation of latent normals
    # constraints values may be proportions in [0,1] OR absolute minimum counts (>1)
    constraints: Dict[str, float]
    N: int = 1000
    max_rejects: int = 20000

    def required_counts(self) -> Dict[str, int]:
        """Return integer minimum counts, supporting either proportions or absolute counts.
        If a value <= 1, it's treated as a proportion of N. If > 1, it's treated as an absolute count.
        """
        req: Dict[str, int] = {}
        for k, v in self.constraints.items():
            if v <= 1.0:
                req[k] = math.ceil(v * self.N)
            else:
                req[k] = int(math.ceil(v))
        return req


# Helper to make a valid correlation matrix (symmetric, PSD-ish) if user tweaks values

def _nearest_psd(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals[vals < eps] = eps
    return (vecs @ np.diag(vals) @ vecs.T)

# ----------------------------
# NEW custom scenarios from the prompt
# ----------------------------

# Helper to build arrays from name->value dicts in a chosen attribute order

def _build_marg_corr(attr_order: List[str], rel_freq: Dict[str, float], corr_map: Dict[str, Dict[str, float]]):
    """Build marginals and correlation matrix in the given order.
    Robust to missing keys by defaulting to 0 off‑diagonal and 1 on the diagonal.
    Also trims/normalizes keys just in case of stray spaces or case differences.
    """
    # Normalize keys (strip spaces) to avoid KeyError due to accidental whitespace
    norm = lambda s: s.strip()
    rel_freq = {norm(k): v for k, v in rel_freq.items()}
    corr_map = {norm(ri): {norm(cj): cv for cj, cv in rj.items()} for ri, rj in corr_map.items()}
    # Fill missing rows/cols with zeros/ones where appropriate
    m = np.array([rel_freq.get(a, 0.5) for a in map(norm, attr_order)], dtype=float)
    C = np.zeros((len(attr_order), len(attr_order)), dtype=float)
    for ii, i in enumerate(map(norm, attr_order)):
        for jj, j in enumerate(map(norm, attr_order)):
            if i == j:
                C[ii, jj] = 1.0
            else:
                C[ii, jj] = corr_map.get(i, {}).get(j, corr_map.get(j, {}).get(i, 0.0))
    return m, C

# Custom Scenario 1
ATTRS_1 = ['young', 'well_dressed']
relfreq_1 = {'well_dressed': 0.3225, 'young': 0.3225}
corrmap_1 = {
    'well_dressed': {'well_dressed': 1.0, 'young': 0.18304299322062992},
    'young': {'well_dressed': 0.18304299322062992, 'young': 1.0},
}
marg_1, corr_1 = _build_marg_corr(ATTRS_1, relfreq_1, corrmap_1)
constraints_1 = {
    'young': 600,          # absolute min counts (since N=1000)
    'well_dressed': 600,
}

# Custom Scenario 2
ATTRS_2 = ['techno_lover', 'well_connected', 'creative', 'berlin_local']
relfreq_2 = {
    'techno_lover': 0.6265000000000001,
    'well_connected': 0.4700000000000001,
    'creative': 0.06227,
    'berlin_local': 0.398,
}
corrmap_2 = {
    'techno_lover': {'techno_lover': 1, 'well_connected': -0.4696169332674324, 'creative': 0.09463317039891586, 'berlin_local': -0.6549403815606182},
    'well_connected': {'techno_lover': -0.4696169332674324, 'well_connected': 1, 'creative': 0.14197259140471485, 'berlin_local': 0.5724067808436452},
    'creative': {'techno_lover': 0.09463317039891586, 'well_connected': 0.14197259140471485, 'creative': 1, 'berlin_local': 0.14446459505650772},
    'berlin_local': {'techno_lover': -0.6549403815606182, 'well_connected': 0.5724067808436452, 'creative': 0.14446459505650772, 'berlin_local': 1},
}
marg_2, corr_2 = _build_marg_corr(ATTRS_2, relfreq_2, corrmap_2)
constraints_2 = {
    'techno_lover': 650,
    'well_connected': 450,
    'creative': 300,
    'berlin_local': 750,
}

# Custom Scenario 3
ATTRS_3 = ['underground_veteran', 'international', 'fashion_forward', 'queer_friendly', 'vinyl_collector', 'german_speaker']
relfreq_3 = {
    'underground_veteran': 0.6794999999999999,
    'international': 0.5735,
    'fashion_forward': 0.6910000000000002,
    'queer_friendly': 0.04614,
    'vinyl_collector': 0.044539999999999996,
    'german_speaker': 0.4565000000000001,
}
corrmap_3 = {
    'underground_veteran': {'underground_veteran': 1, 'international': -0.08110175777152992, 'fashion_forward': -0.1696563475505309, 'queer_friendly': 0.03719928376753885, 'vinyl_collector': 0.07223521156389842, 'german_speaker': 0.11188766703422799},
    'international': {'underground_veteran': -0.08110175777152992, 'international': 1, 'fashion_forward': 0.375711059360155, 'queer_friendly': 0.0036693314388711686, 'vinyl_collector': -0.03083247098181075, 'german_speaker': -0.7172529382519395},
    'fashion_forward': {'underground_veteran': -0.1696563475505309, 'international': 0.375711059360155, 'fashion_forward': 1, 'queer_friendly': -0.0034530926793377476, 'vinyl_collector': -0.11024719606358546, 'german_speaker': -0.3521024461597403},
    'queer_friendly': {'underground_veteran': 0.03719928376753885, 'international': 0.0036693314388711686, 'fashion_forward': -0.0034530926793377476, 'queer_friendly': 1, 'vinyl_collector': 0.47990640803167306, 'german_speaker': 0.04797381132680503},
    'vinyl_collector': {'underground_veteran': 0.07223521156389842, 'international': -0.03083247098181075, 'fashion_forward': -0.11024719606358546, 'queer_friendly': 0.47990640803167306, 'german_speaker': 0.09984452286269897},
    'german_speaker': {'underground_veteran': 0.11188766703422799, 'international': -0.7172529382519395, 'fashion_forward': -0.3521024461597403, 'queer_friendly': 0.04797381132680503, 'vinyl_collector': 0.09984452286269897, 'german_speaker': 1},
}
marg_3, corr_3 = _build_marg_corr(ATTRS_3, relfreq_3, corrmap_3)
constraints_3 = {
    'underground_veteran': 500,
    'international': 650,
    'fashion_forward': 550,
    'queer_friendly': 250,
    'vinyl_collector': 200,
    'german_speaker': 800,
}

SCENARIOS: Dict[str, Scenario] = {
    'custom_scenario_1': Scenario('custom_scenario_1', ATTRS_1, marg_1, _nearest_psd(corr_1), constraints_1),
    'custom_scenario_2': Scenario('custom_scenario_2', ATTRS_2, marg_2, _nearest_psd(corr_2), constraints_2),
    'custom_scenario_3': Scenario('custom_scenario_3', ATTRS_3, marg_3, _nearest_psd(corr_3), constraints_3),
}


# ----------------------------
# Gaussian‑copula Bernoulli sampler
# ----------------------------

class CorrelatedBernoulli:
    def __init__(self, marginals: np.ndarray, corr: np.ndarray, rng: np.random.Generator):
        self.m = np.asarray(marginals).astype(float)
        self.K = self.m.shape[0]
        self.C = _nearest_psd(np.asarray(corr).astype(float))
        # Cholesky with jitter for numerical stability
        jitter = 1e-10
        for _ in range(5):
            try:
                self.L = np.linalg.cholesky(self.C + jitter * np.eye(self.K))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            # final fallback (eigendecomp)
            vals, vecs = np.linalg.eigh(self.C)
            vals[vals < 1e-9] = 1e-9
            self.L = np.linalg.cholesky(vecs @ np.diag(vals) @ vecs.T)
        self.rng = rng
        # Thresholds for probit transform
        from math import sqrt
        from math import erf
        def inv_norm_cdf(p: float) -> float:
            # Approximate inverse CDF via binary search on erf-based CDF
            lo, hi = -8.0, 8.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                cdf = 0.5 * (1.0 + erf(mid / sqrt(2)))
                if cdf < p:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        self.thresholds = np.array([inv_norm_cdf(pi) for pi in self.m])

    def sample(self) -> np.ndarray:
        z = self.rng.normal(size=self.K)
        x = self.L @ z
        return (x < self.thresholds).astype(np.int8)


# ----------------------------
# Environment
# ----------------------------

class BerghainEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, scenario: Scenario, seed: Optional[int] = None):
        super().__init__()
        self.scenario = scenario
        self.N = scenario.N
        self.max_rejects = scenario.max_rejects
        self.attrs = scenario.attrs
        self.K = len(self.attrs)
        self.action_space = spaces.Discrete(2)  # 0=reject, 1=accept
        self.observation_space = spaces.Dict(
            {
                'attrs': spaces.MultiBinary(self.K),
                'accepted': spaces.Discrete(self.N + 1),
                'rejected': spaces.Discrete(self.max_rejects + 1),
            }
        )
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._sampler = CorrelatedBernoulli(scenario.marginals, scenario.corr, self._rng)

        # State
        self.accepted = 0
        self.rejected = 0
        self.counts = np.zeros(self.K, dtype=int)  # counts among accepted only
        self.current = self._sampler.sample()

    # Gymnasium compatibility
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            self._sampler = CorrelatedBernoulli(self.scenario.marginals, self.scenario.corr, self._rng)
        self.accepted = 0
        self.rejected = 0
        self.counts = np.zeros(self.K, dtype=int)
        self.current = self._sampler.sample()
        return self._obs(), self._info()

    def _obs(self):
        return {
            'attrs': self.current.copy(),
            'accepted': int(self.accepted),
            'rejected': int(self.rejected),
        }

    def _info(self):
        return {
            'attr_names': list(self.attrs),
            'required_counts': self.scenario.required_counts(),
            'counts': {name: int(self.counts[i]) for i, name in enumerate(self.attrs)},
        }

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        if action not in (0, 1):
            raise ValueError("Action must be 0 (reject) or 1 (accept)")

        if action == 1:
            # accept
            self.accepted += 1
            self.counts += self.current
        else:
            # reject
            self.rejected += 1

        # draw next person only if not done
        if self.accepted < self.N and self.rejected < self.max_rejects:
            self.current = self._sampler.sample()
        
        # check termination
        if self.accepted >= self.N or self.rejected >= self.max_rejects:
            terminated = True
            success = self._constraints_satisfied() and (self.accepted >= self.N)
            if success:
                reward = -float(self.rejected)
            else:
                reward = -100000.0
            info = {
                **self._info(),
                'success': success,
                'accepted': int(self.accepted),
                'rejected': int(self.rejected),
            }
        return self._obs(), reward, terminated, truncated, info

    def _constraints_satisfied(self) -> bool:
        req = self.scenario.required_counts()
        for name, min_count in req.items():
            idx = self.attrs.index(name)
            if self.counts[idx] < min_count:
                return False
        return True

    def render(self):
        req = self.scenario.required_counts()
        print(f"Accepted: {self.accepted}  Rejected: {self.rejected}")
        for i, name in enumerate(self.attrs):
            have = self.counts[i]
            need = req.get(name, 0)
            print(f"  {name:16s}: {have} / {need}")
        print(f"Current visitor: {dict(zip(self.attrs, self.current.tolist()))}")


def make_env(scenario_key: str = 'scenario_a', seed: Optional[int] = None) -> BerghainEnv:
    if scenario_key not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_key}'. Options: {list(SCENARIOS)}")
    return BerghainEnv(SCENARIOS[scenario_key], seed=seed)


# ----------------------------
# Feasibility‑aware baseline policy
# ----------------------------

class FeasibilityBaseline:
    """
    Myopic policy: ensure it's still possible to meet all constraints with remaining slots.
    Accepts an applicant if either
      (a) rejecting them would make some constraint impossible; or
      (b) we are already on track and can accept without harming feasibility;
    and prefers accepting those that help satisfy the tightest constraints.
    """
    def __init__(self, env: BerghainEnv):
        self.env = env
        self.req = env.scenario.required_counts()
        self.attr_to_idx = {name: i for i, name in enumerate(env.attrs)}

    def act(self, obs) -> int:
        # Extract
        x = np.array(obs['attrs'], dtype=int)
        A = self.env.accepted
        N = self.env.N
        S = N - A  # slots remaining if we reject; if accept, S_after = N - (A+1)
        counts = self.env.counts

        # Check for must‑accept cases: if rejecting this person (who has attr) would make meeting requirement impossible
        for name, min_count in self.req.items():
            i = self.attr_to_idx[name]
            have = counts[i]
            need = min_count
            feasible_if_reject = (have + S) >= need
            if (not feasible_if_reject) and (x[i] == 1):
                return 1  # accept

        # Check for must‑reject cases: if accepting a non‑qualifying person would make some constraint impossible
        S_after = S - 1
        for name, min_count in self.req.items():
            i = self.attr_to_idx[name]
            have_if_accept = counts[i] + int(x[i] == 1)
            feasible_if_accept = (have_if_accept + S_after) >= min_count
            if not feasible_if_accept:
                return 0  # must reject

        # Otherwise, heuristic preference: accept if they satisfy the most constrained requirements
        score = 0.0
        for name, min_count in self.req.items():
            i = self.attr_to_idx[name]
            gap = min_count - counts[i]
            if x[i] == 1 and gap > 0:
                score += 1 + 1.0 / (1 + gap)
        if score > 0:
            return 1
        return 1 if np.random.random() < 0.25 else 0


# ----------------------------
# Quick self‑test (includes the three new scenarios)
# ----------------------------

def _quick_run(env_key='custom_scenario_1', episodes=1, seed=0):
    env = make_env(env_key, seed=seed)
    pi = FeasibilityBaseline(env)
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done:
            a = pi.act(obs)
            obs, rew, term, trunc, info = env.step(a)
            done = term or trunc
            steps += 1
            if steps > (env.N + env.max_rejects + 5):
                raise RuntimeError("Episode too long; check termination conditions.")
        results.append((rew, info.get('success', False), info.get('accepted'), info.get('rejected')))
    return results


if __name__ == '__main__':
    # Smoke test of the custom scenarios
    for key in ['custom_scenario_1', 'custom_scenario_2', 'custom_scenario_3']:
        res = _quick_run(key, episodes=2, seed=123)
        print(key, res)
