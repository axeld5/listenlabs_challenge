"""
Gymnasium environment for the Berghain Challenge.

This environment simulates the nightclub bouncer decision problem where you must
accept or reject visitors to meet various constraints while minimizing rejections.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # fallback to classic gym if needed
    import gym
    from gym import spaces

from scenarios import Scenario, SCENARIOS
from sampler import CorrelatedBernoulli


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
        self.observation_space = spaces.Dict({
            'attrs': spaces.MultiBinary(self.K),
            'accepted': spaces.Discrete(self.N + 1),
            'rejected': spaces.Discrete(self.max_rejects + 1),
            'remaining': spaces.MultiDiscrete([self.N + 1] * self.K),  # NEW
        })
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
        rem = [self._remaining_needs().get(name, 0) for name in self.attrs]
        return {
            'attrs': self.current.copy(),
            'accepted': int(self.accepted),
            'rejected': int(self.rejected),
            'remaining': np.array(rem, dtype=np.int32),
        }

    def _impossible_now(self) -> bool:
        remaining_slots = self.N - self.accepted
        for need in self._remaining_needs().values():
            if remaining_slots < need:
                return True
        return False
    
    def _remaining_needs(self) -> Dict[str, int]:
        req = self.scenario.required_counts()
        rem = {}
        for name, need in req.items():
            idx = self.attrs.index(name)
            have = int(self.counts[idx])
            rem[name] = max(0, need - have)
        return rem

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

        if not terminated and self._impossible_now():
            terminated = True
            # big fail, but not astronomical; keep gradient sane
            reward = -100.0

        # check termination
        if self.accepted >= self.N or self.rejected >= self.max_rejects:
            terminated = True
            success = self._constraints_satisfied() and (self.accepted >= self.N)
            if success:
                reward = -float(self.rejected)/10000
            else:
                reward = -100.0
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
