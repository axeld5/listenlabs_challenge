"""
Gaussian-copula Bernoulli sampler for generating correlated binary attributes.
"""
from __future__ import annotations

import math

import numpy as np

from scenarios import _nearest_psd


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
        def inv_norm_cdf(p: float) -> float:
            # Approximate inverse CDF via binary search on erf-based CDF
            lo, hi = -8.0, 8.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2)))
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
