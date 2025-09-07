"""
SB3-based PPO policy for the Berghain Challenge environment.

This policy uses Stable-Baselines3's battle-tested PPO implementation
for reliable reinforcement learning on the Berghain Challenge.
"""
from __future__ import annotations

from typing import Optional
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from wrappers import FloatifyObs


from environment import BerghainEnv


class PPOCallback(BaseCallback):
    """Custom callback for PPO training progress."""

    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            avg_reward, success_rate = self._evaluate_policy()
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
        return True

    def _evaluate_policy(self, eval_episodes: int = 5) -> tuple[float, float]:
        """Evaluate current policy performance."""
        env = self.training_env.envs[0]  # Get the underlying environment
        total_reward = 0
        successes = 0

        for ep in range(eval_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward
            if info.get('success', False):
                successes += 1

        avg_reward = total_reward / eval_episodes
        success_rate = successes / eval_episodes
        return avg_reward, success_rate


class SB3PPOPolicy:
    def __init__(self, env: BerghainEnv, policy_path: Optional[str] = None,
                 learning_rate: float = 3e-4, n_steps: int = 2048,
                 batch_size: int = 64, n_epochs: int = 10,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_range: float = 0.2, ent_coef: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 verbose: int = 1, device: str = "auto",
                 n_envs: int = 8, seed: int = 0, **kwargs):

        self.env = env
        self._seed = seed

        def make_one(rank: int):
            def _init():
                e = BerghainEnv(env.scenario, seed=None)   # fresh env
                e = FloatifyObs(e)                         # cast counts to float Boxes
                e.reset(seed=seed + rank)
                return e
            return _init

        set_random_seed(seed)
        train_vec = SubprocVecEnv([make_one(i) for i in range(n_envs)])
        train_vec = VecMonitor(train_vec)  # single VecMonitor at vector level

        # Only normalize the Box keys (attrs stays MultiBinary in the base env)
        self.monitor_env = VecNormalize(
            train_vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            norm_obs_keys=["accepted", "rejected", "remaining"],  # <-- important
        )

        # PPO init as before...
        ppo_kwargs = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm,
            'verbose': verbose,
            'device': device,
            **kwargs
        }
        policy_name = "MultiInputPolicy"
        if policy_path is not None and os.path.exists(policy_path):
            print(f"Loading PPO model from {policy_path}")
            self.model = PPO.load(policy_path, env=self.monitor_env, **ppo_kwargs)
        else:
            print("Creating new PPO model (parallel envs + normalization)")
            self.model = PPO(policy_name, self.monitor_env, **ppo_kwargs)

    def _prepare_single_obs(self, obs: dict) -> dict:
            """Match the training-time FloatifyObs shapes/dtypes for a single observation."""
            return {
                'attrs': np.asarray(obs['attrs'], dtype=np.int8),
                'accepted': np.array([obs['accepted']], dtype=np.float32),  # (1,)
                'rejected': np.array([obs['rejected']], dtype=np.float32),  # (1,)
                'remaining': np.asarray(obs['remaining'], dtype=np.float32) # (K,)
            }

    def _prepare_single_obs(self, obs: dict) -> dict:
        import numpy as np
        return {
            'attrs': np.asarray(obs['attrs'], dtype=np.int8),
            'accepted': np.array([obs['accepted']], dtype=np.float32),  # (1,)
            'rejected': np.array([obs['rejected']], dtype=np.float32),  # (1,)
            'remaining': np.asarray(obs['remaining'], dtype=np.float32) # (K,)
        }

    def act(self, obs) -> int:
        import numpy as np
        # shape-correct the raw dict obs
        if isinstance(obs, dict):
            obs = self._prepare_single_obs(obs)
            # add batch dim for VecNormalize
            batched = {k: v[None, ...] for k, v in obs.items()}
            # normalize using running stats (only the keys you set in norm_obs_keys)
            norm = self.monitor_env.normalize_obs(batched)  # <-- no clip_obs kw
            # remove batch dim
            obs = {k: v[0] for k, v in norm.items()}

        action, _ = self.model.predict(obs, deterministic=True)
        return int(action.item() if isinstance(action, np.ndarray) else action)



    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000, progress_bar: bool = True) -> PPO:
        print(f"Starting PPO training for {total_timesteps} timesteps...")

        def make_eval_env():
            e = BerghainEnv(self.env.scenario, seed=12345)
            e = FloatifyObs(e)
            return e

        eval_raw = DummyVecEnv([make_eval_env])
        eval_raw = VecMonitor(eval_raw)

        eval_vec = VecNormalize(
            eval_raw,
            training=False,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            norm_obs_keys=["accepted", "rejected", "remaining"],
        )
        # share running stats so eval uses identical normalization
        eval_vec.obs_rms = self.monitor_env.obs_rms
        eval_vec.ret_rms = self.monitor_env.ret_rms

        eval_cb = EvalCallback(
            eval_vec,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        ) if eval_freq and eval_freq > 0 else None

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_cb,
            progress_bar=progress_bar
        )
        print("Training completed!")
        return self.model


    def save(self, path: str):
        self.model.save(path)
        # save VecNormalize stats next to the model
        self.monitor_env.save(path + "_vecnorm.pkl")
        print(f"Model saved to {path} (+ normalization stats)")

    def load(self, path: str):
        # the constructor already created self.monitor_env; just load stats into it
        self.monitor_env = VecNormalize.load(path + "_vecnorm.pkl", self.monitor_env)
        self.model = PPO.load(path, env=self.monitor_env)
        print(f"Model + normalization loaded from {path}")

    def evaluate(self, eval_episodes: int = 10) -> tuple[float, float]:
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_eval_env():
            e = BerghainEnv(self.env.scenario, seed=98765)
            e = FloatifyObs(e)
            e = Monitor(e)
            return e

        eval_raw = DummyVecEnv([make_eval_env])
        eval_vec = VecNormalize(eval_raw, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0)
        # share current stats
        eval_vec.obs_rms = self.monitor_env.obs_rms
        eval_vec.ret_rms = self.monitor_env.ret_rms

        total_reward, successes = 0.0, 0
        for _ in range(eval_episodes):
            obs = eval_vec.reset()
            done = False
            ep_rew = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_vec.step(action)
                ep_rew += float(reward)
                done = bool(terminated) or bool(truncated)
            total_reward += ep_rew
            # success flag is inside info list for vec envs
            if info and isinstance(info, (list, tuple)) and len(info) and info[0].get('success', False):
                successes += 1

        return total_reward / eval_episodes, successes / eval_episodes



# Backward compatibility - create alias for old interface
PPOPolicy = SB3PPOPolicy
FeasibilityBaseline = SB3PPOPolicy


# Example usage:
"""
Example of how to use the SB3 PPO policy:

from environment import make_env
from baseline_policy import SB3PPOPolicy

# Create environment
env = make_env('custom_scenario_1', seed=42)

# Create PPO policy with SB3
policy = SB3PPOPolicy(env,
    learning_rate=3e-4,      # Learning rate
    n_steps=2048,            # Steps per environment per update
    batch_size=64,           # Minibatch size
    n_epochs=10,             # Number of epochs per update
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_range=0.2,          # PPO clipping range
    ent_coef=0.01,           # Entropy coefficient
    vf_coef=0.5,             # Value function coefficient
    verbose=1                # Verbosity level
)

# Train the policy
policy.train(total_timesteps=50000, eval_freq=5000)

# Evaluate the trained policy
avg_reward, success_rate = policy.evaluate(eval_episodes=10)
print(f"Average reward: {avg_reward:.2f}")
print(f"Success rate: {success_rate:.2f}")

# Save the trained policy
policy.save("sb3_ppo_policy.zip")

# Load a pre-trained policy
# policy = SB3PPOPolicy(env, policy_path="sb3_ppo_policy.zip")

# Use policy for inference
obs, info = env.reset()
done = False
while not done:
    action = policy.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
"""
