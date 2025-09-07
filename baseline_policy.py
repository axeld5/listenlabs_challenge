from __future__ import annotations

import numpy as np

from environment import BerghainEnv

# --- Heuristic teacher policy (safe feasibility) ---
class FeasibilityBaseline:
    """
    Myopic policy that preserves feasibility.
    Accept if (a) rejecting would make a constraint impossible; or
    (b) accepting does not break feasibility and helps the tightest gaps.
    """
    def __init__(self, env: BerghainEnv):
        self.env = env
        self.req = env.scenario.required_counts()
        self.attr_to_idx = {name: i for i, name in enumerate(env.attrs)}

    def act(self, obs) -> int:
        x = np.array(obs['attrs'], dtype=int)
        A = self.env.accepted
        N = self.env.N
        S = N - A                 # slots remaining if we reject
        counts = self.env.counts

        # (1) Must-accept: rejecting would make some constraint impossible AND this person has that attr
        for name, need in self.req.items():
            i = self.attr_to_idx[name]
            have = int(counts[i])
            feasible_if_reject = (have + S) >= need
            if (not feasible_if_reject) and (x[i] == 1):
                return 1

        # (2) Must-reject: accepting would make some constraint impossible
        S_after = S - 1
        for name, need in self.req.items():
            i = self.attr_to_idx[name]
            have_if_accept = counts[i] + int(x[i] == 1)
            feasible_if_accept = (have_if_accept + S_after) >= need
            if not feasible_if_accept:
                return 0

        # (3) Prefer accepting those that help the tightest gaps
        score = 0.0
        for name, need in self.req.items():
            i = self.attr_to_idx[name]
            gap = need - counts[i]
            if x[i] == 1 and gap > 0:
                score += 1.0 + 1.0 / (1.0 + gap)
        if score > 0:
            return 1

        # (4) Mild exploration to avoid deadlocks
        return 1 if np.random.random() < 0.25 else 0
