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
from baseline_policy import PPOPolicy


# ----------------------------
# Quick self‑test (includes the three new scenarios)
# ----------------------------

def _quick_run(env_key='custom_scenario_1', episodes=1, seed=0, train_timesteps=10000):
    env = make_env(env_key, seed=seed)
    pi = PPOPolicy(env, verbose=0)

    # Train the policy first
    print(f"Training SB3 PPO policy for {train_timesteps} timesteps...")
    pi.train(total_timesteps=train_timesteps, eval_freq=0)  # Disable callback for now

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
    for key in ['custom_scenario_1']: #, 'custom_scenario_2', 'custom_scenario_3']:
        res = _quick_run(key, episodes=2, seed=123, train_timesteps=5000)  # Reduced for quick testing
        print(key, res)
