"""
Scenario configuration for Berghain Challenge environments.

Contains scenario definitions, helper functions for building marginals and correlations,
and the main SCENARIOS dictionary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


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
    'vinyl_collector': {'underground_veteran': 0.07223521156389842, 'international': -0.03083247098181075, 'fashion_forward': -0.11024719606358546, 'queer_friendly': 0.47990640803167306, 'vinyl_collector': 1, 'german_speaker': 0.09984452286269897},
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
