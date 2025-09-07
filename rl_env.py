"""
Berghain Challenge — Multi-turn RL Environment (custom scenarios)

You are the bouncer at a nightclub. People arrive one by one with binary attributes.
Decide accept(1)/reject(0) immediately. Goal: fill the venue with N people while
meeting all minimum proportion/absolute constraints. Minimize rejections.

API: Gymnasium-style environment.

- Observation: Dict with
    - 'attrs': MultiBinary(K) — current visitor's attributes (0/1)
    - 'accepted': Discrete(N+1)
    - 'rejected': Discrete(max_rejects+1)
- Action: Discrete(2) — 0=reject, 1=accept
- Termination:
    - Success: accepted == N and constraints satisfied  -> reward = -rejected_count
    - Failure: accepted == N and constraints NOT satisfied OR rejected == max_rejects -> reward = -100000

Three built-in baseline scenarios remain, plus three new custom scenarios from the prompt.
Attributes are sampled i.i.d. with specified marginals and correlations via a Gaussian copula.

Usage
-----
from rl_env import BerghainEnv, make_env, SCENARIOS

env = make_env('custom_scenario_1', seed=42)
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(reward, info)

See bottom for example baseline policies and quick self-test.
"""
from __future__ import annotations

from environment import make_env
from ppo_policy import PPOPolicy

import torch.nn as nn
from pathlib import Path

from baseline_policy import FeasibilityBaseline  # import the heuristic
from environment import BerghainEnv

def warm_safe_run(env_key='custom_scenario_1', seed=0,
                  bc_episodes=400, bc_epochs=5, bc_batch=4096,
                  total_timesteps=2_000_000):
    env = make_env(env_key, seed=seed)

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        activation_fn=nn.ReLU,
        ortho_init=True,
    )

    pi = PPOPolicy(
        env,
        verbose=1,
        seed=seed,
        n_envs=16,
        n_steps=8192,
        batch_size=2048,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
    )

    teacher = FeasibilityBaseline(BerghainEnv(env.scenario, seed=seed))
    print("BC warm-start from FeasibilityBaseline...")
    pi.pretrain_with_teacher(teacher=teacher, episodes=bc_episodes, epochs=bc_epochs, batch_size=bc_batch, seed=seed)

    print(f"PPO fine-tuning for {total_timesteps} timesteps...")
    pi.train(total_timesteps=total_timesteps, eval_freq=50_000, progress_bar=True)

    Path("models").mkdir(exist_ok=True)
    save_path = f"models/{env_key}_warm_safe_ppo.zip"
    pi.save(save_path)

    avg_reward, success_rate = pi.evaluate(eval_episodes=20)
    print(f"[WARM+SAFE] avg_reward: {avg_reward:.3f} | success_rate: {success_rate:.3f}")
    print(f"Saved to: {save_path} and {save_path}_vecnorm.pkl")
    return save_path, (avg_reward, success_rate)


# ----------------------------
# Quick self‑test (includes the three new scenarios)
# ----------------------------

def print_policy_arch(pi):
    # pi is your SB3PPOPolicy instance
    policy = pi.model.policy  # sb3 torch.nn.Module
    print("\n=== Policy Module ===")
    print(policy)  # full module tree

    # Useful submodules (MultiInputPolicy)
    try:
        print("\n--- Features Extractor ---")
        print(policy.features_extractor)

        print("\n--- MLP Extractor (shared trunk) ---")
        print(policy.mlp_extractor)

        print("\n--- Actor (pi) head ---")
        print(policy.mlp_extractor.policy_net)

        print("\n--- Critic (vf) head ---")
        print(policy.mlp_extractor.value_net)
    except Exception as e:
        print(f"(Could not dig into submodules: {e})")

import os
from pathlib import Path
import torch.nn as nn

def long_run(env_key='custom_scenario_1', seed=0, total_timesteps=3_000_000):
    env = make_env(env_key, seed=seed)

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        activation_fn=nn.ReLU,
        ortho_init=True,
    )

    pi = PPOPolicy(
        env,
        verbose=1,
        seed=seed,
        n_envs=16,
        n_steps=8192,
        batch_size=2048,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
    )

    print_policy_arch(pi)  # show the larger architecture

    print(f"Starting LONG RUN for {total_timesteps} timesteps...")
    pi.train(total_timesteps=total_timesteps, eval_freq=50_000, progress_bar=True)

    Path("models").mkdir(exist_ok=True)
    save_path = f"models/{env_key}_longrun_ppo.zip"
    pi.save(save_path)

    avg_reward, success_rate = pi.evaluate(eval_episodes=20)
    print(f"\n[LONG RUN] avg_reward: {avg_reward:.3f} | success_rate: {success_rate:.3f}")
    print(f"Saved to: {save_path} and {save_path}_vecnorm.pkl")
    return save_path, (avg_reward, success_rate)

def _quick_run(env_key='custom_scenario_1', episodes=1, seed=0, train_timesteps=10000):
    env = make_env(env_key, seed=seed)
    pi = PPOPolicy(env, verbose=0)

    # Train the policy first
    print_policy_arch(pi)  # print immediately (untrained)
    print(f"Training SB3 PPO policy for {train_timesteps} timesteps...")
    pi.train(total_timesteps=train_timesteps, eval_freq=0)

    print("\n=== Policy Module (after training) ===")

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'long', 'warmlong', 'warm_safe'], default='quick')
    parser.add_argument('--scenario', default='custom_scenario_1')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--timesteps', type=int, default=5_000)  # for quick
    args = parser.parse_args()

    if args.mode == 'quick':
        res = _quick_run(args.scenario, episodes=2, seed=args.seed, train_timesteps=args.timesteps)
        print(args.scenario, res)
    elif args.mode == 'warm_safe':
        warm_safe_run(args.scenario, seed=args.seed, bc_episodes=400, total_timesteps=2_000_000)
    else:
        # keep your other modes if you added them earlier
        warm_safe_run(args.scenario, seed=args.seed, bc_episodes=0, total_timesteps=3_000_000)