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
    """
    PPO-based policy using Stable-Baselines3 for the Berghain Challenge environment.
    """

    def __init__(self, env: BerghainEnv, policy_path: Optional[str] = None,
                 learning_rate: float = 3e-4, n_steps: int = 2048,
                 batch_size: int = 64, n_epochs: int = 10,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_range: float = 0.2, ent_coef: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 verbose: int = 1, device: str = "auto", **kwargs):

        self.env = env

        # Wrap environment with Monitor and DummyVecEnv for SB3 compatibility
        def make_env():
            return Monitor(env)

        self.monitor_env = DummyVecEnv([make_env])

        # PPO hyperparameters
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

        # Initialize PPO model
        if policy_path is not None and os.path.exists(policy_path):
            print(f"Loading PPO model from {policy_path}")
            self.model = PPO.load(policy_path, env=self.monitor_env, **ppo_kwargs)
        else:
            print("Creating new PPO model")
            # Use MultiInputPolicy for dict observation spaces
            policy_name = "MultiInputPolicy"
            self.model = PPO(policy_name, self.monitor_env, **ppo_kwargs)

    def act(self, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        # handle shape (1,) vs scalar
        if isinstance(action, np.ndarray):
            return int(action.item())
        return int(action)

    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000,
              progress_bar: bool = True) -> PPO:
        """
        Train the PPO policy.

        Args:
            total_timesteps: Total number of environment steps to train for
            eval_freq: Frequency of evaluation callbacks
            progress_bar: Whether to show progress bar
        """
        print(f"Starting PPO training for {total_timesteps} timesteps...")

        # Create callback for evaluation
        callback = PPOCallback(eval_freq=eval_freq, verbose=1) if eval_freq > 0 else None

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar
        )

        print("Training completed!")
        return self.model

    def save(self, path: str):
        """Save the trained PPO model."""
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load a trained PPO model."""
        self.model = PPO.load(path, env=self.monitor_env)
        print(f"Model loaded from {path}")

    def evaluate(self, eval_episodes: int = 10) -> tuple[float, float]:
        """Evaluate the current policy."""
        total_reward = 0
        successes = 0

        for ep in range(eval_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.act(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward
            if info.get('success', False):
                successes += 1

        avg_reward = total_reward / eval_episodes
        success_rate = successes / eval_episodes
        return avg_reward, success_rate


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
