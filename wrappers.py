from gymnasium import ObservationWrapper, spaces
import numpy as np

class FloatifyObs(ObservationWrapper):
    """
    Cast integer-like observations (accepted, rejected, remaining) to float32 so VecNormalize can work.
    """
    def __init__(self, env):
        super().__init__(env)
        orig = env.observation_space
        self.observation_space = spaces.Dict({
            'attrs': orig['attrs'],  # keep binary as is (SB3 flattens)
            'accepted': spaces.Box(0, env.N, shape=(1,), dtype=np.float32),
            'rejected': spaces.Box(0, env.max_rejects, shape=(1,), dtype=np.float32),
            'remaining': spaces.Box(0, env.N, shape=(len(env.attrs),), dtype=np.float32),
        })

    def observation(self, obs):
        return {
            'attrs': obs['attrs'],
            'accepted': np.array([obs['accepted']], dtype=np.float32),
            'rejected': np.array([obs['rejected']], dtype=np.float32),
            'remaining': obs['remaining'].astype(np.float32),
        }