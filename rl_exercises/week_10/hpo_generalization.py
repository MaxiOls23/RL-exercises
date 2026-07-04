from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_exercises.week_9.dyna_ppo import DynaPPOAgent

SEARCH_SPACE = {
    "imag_horizon": (1, 15),
    "imag_batches": (1, 15),
    "model_epochs": (1, 5),
}

SCORE_ALPHA = 0.6

DEFAULT_CONFIG = dict( # default config from last weeks experiment
    # PPO
    epochs=4, batch_size=64,
    lr_actor=3e-4, lr_critic=1e-3,
    gamma=0.99, gae_lambda=0.95,
    clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
    hidden_size=64,
    # Dyna
    use_model=True,
    model_lr=1e-3, model_epochs=3, model_batch_size=64,
    imag_horizon=5, imag_batches=10, max_buffer_size=10000,
)


def evaluate_config(
        config: Dict[str, int],
        seeds: List[int],
        total_steps: int,
        eval_interval: int = 1000,
) -> List[float]:
    finals = []
    for seed in seeds:
        agent = make_agent(seed=seed, **config)
        metrics = train_and_collect(
            agent, total_steps=total_steps, eval_interval=eval_interval, seed=seed
        )
        finals.append(float(np.mean(metrics["returns"][-3:])) if metrics["returns"] else 0.0)
    return finals


def make_agent(seed: int = 42, **overrides) -> DynaPPOAgent:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    cfg = {**DEFAULT_CONFIG, "seed": seed}
    cfg.update(overrides)

    return DynaPPOAgent(env, **cfg)


def train_and_collect(
        agent: DynaPPOAgent,
        total_steps: int,
        eval_interval: int = 1000,
        eval_episodes: int = 5,
        seed: int = 42,
) -> Dict[str, List]:
    eval_env = gym.make(agent.env.spec.id)
    eval_env.reset(seed=seed + 1)

    metrics: Dict[str, List] = {
        "real_steps": [], "returns": [],
        "model_state_mse": [], "model_reward_mse": [],
    }

    while agent.real_steps < total_steps:
        state, _ = agent.env.reset()
        done = False
        real_traj: List[Any] = []

        while not done and agent.real_steps < total_steps:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = agent.policy(s_t)
                dist = torch.distributions.Categorical(logits=logits)
                a_t = dist.sample()
                logp_t = dist.log_prob(a_t)
                ent_t = dist.entropy()

            action = a_t.item()
            next_state, reward, term, trunc, _ = agent.env.step(action)
            done = term or trunc
            real_traj.append(
                (state, action, logp_t.squeeze(), ent_t.squeeze(),
                 float(reward), float(done), next_state)
            )
            state = next_state
            agent.real_steps += 1

            if agent.real_steps % eval_interval == 0:
                mean_r, _ = agent.evaluate(eval_env, num_episodes=eval_episodes)
                metrics["real_steps"].append(agent.real_steps)
                metrics["returns"].append(mean_r)

                n_eval = min(500, len(agent.real_buffer)) if agent.use_model else 0
                if agent.use_model and n_eval >= max(5, agent.model_batch_size):
                    m = agent.evaluate_model(num_samples=n_eval)
                else:
                    m = {"state_mse": 0.0, "reward_mse": 0.0}
                metrics["model_state_mse"].append(m["state_mse"])
                metrics["model_reward_mse"].append(m["reward_mse"])

        agent.total_episodes += 1
        agent.update(real_traj)

        if agent.use_model:
            agent.store_real(real_traj)
            agent.train_model()
            agent.imagine_and_update()

    return metrics


def run_gp_bo(opt_seeds, total_steps, n_trials, n_init: int = 3, seed: int = 0) -> Dict[str, int]:
    """Run Gaussian-process Bayesian optimization.

    Randomly evaluates an initial set of configurations before fitting a
    Gaussian process surrogate. New candidates are selected using the
    Expected Improvement acquisition function.

    Args:
        opt_seeds: Training seeds used to estimate each configuration's quality.
        total_steps: Number of environment steps used for each evaluation run.
        n_trials: Total number of configurations to evaluate.
        n_init: Number of random warm-up evaluations before GP-guided search.
        seed: Random seed controlling candidate sampling inside the optimizer.

    Returns:
        The best configuration found, mapped back to the discrete search space.
    """

    rng = np.random.default_rng(seed)
    names = list(SEARCH_SPACE.keys())
    bounds = np.array([SEARCH_SPACE[n] for n in names], dtype=float)

    def sample_random() -> np.ndarray:
        return rng.integers(bounds[:, 0], bounds[:, 1] + 1).astype(float)

    def to_cfg(x: np.ndarray) -> Dict[str, int]:
        return {n: int(round(v)) for n, v in zip(names, x)}

    configs = []
    scores = []

    n_init = min(n_init, n_trials)
    for _ in range(n_init):
        x = sample_random()
        vals = evaluate_config(to_cfg(x), opt_seeds, total_steps)
        score = float(np.mean(vals) - SCORE_ALPHA * np.std(vals))
        configs.append(x)
        scores.append(score)

    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, alpha=1e-6)

    for _ in range(n_trials - n_init):
        gp.fit(np.array(configs), np.array(scores))

        candidates = np.array([sample_random() for _ in range(200)])
        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        best_y = max(scores)
        imp = mu - best_y
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)

        x_next = candidates[int(np.argmax(ei))]
        vals_next = evaluate_config(to_cfg(x_next), opt_seeds, total_steps)
        score = float(np.mean(vals_next) - SCORE_ALPHA * np.std(vals_next))
        configs.append(x_next)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return to_cfg(configs[best_idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Level 1: HPO generalization across seeds")
    parser.add_argument("--opt_seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Seeds used DURING Bayesian Optimization")
    parser.add_argument("--test_seeds", type=int, nargs="+", default=[10, 11],
                        help="Held-out seeds, never seen by BO, used to test generalization")
    parser.add_argument("--total_steps", type=int, default=1000,
                        help="Real env steps per training run")
    parser.add_argument("--n_trials", type=int, default=2,
                        help="Number of BO trials")
    parser.add_argument("--output_dir", default="results")

    args = parser.parse_args()

    overlap = set(args.opt_seeds) & set(args.test_seeds)
    if overlap:
        raise ValueError(f"opt_seeds and test_seeds must be disjoint, overlap: {overlap}")

    print(f"Optimization seeds : {args.opt_seeds}")
    print(f"Held-out seeds     : {args.test_seeds}")
    print(f"Total steps/run    : {args.total_steps}")
    print(f"BO trials          : {args.n_trials}")

    print("\n[1/3] Running Bayesian Optimization on optimization seeds ...")
    best_cfg = run_gp_bo(args.opt_seeds, args.total_steps, args.n_trials)
    print(f"  Best config found: {best_cfg}")

    print("\n[2/3] Evaluating BO config on opt-seeds (in-sample) and test-seeds (held-out) ...")
    bo_opt_scores = evaluate_config(best_cfg, args.opt_seeds, args.total_steps)
    bo_test_scores = evaluate_config(best_cfg, args.test_seeds, args.total_steps)

    print("\n[3/3] Evaluating Level-2 DEFAULT config on the same seed splits ...")
    default_overrides = {
        k: v for k, v in DEFAULT_CONFIG.items() if k in SEARCH_SPACE
    }
    base_opt_scores = evaluate_config(default_overrides, args.opt_seeds, args.total_steps)
    base_test_scores = evaluate_config(default_overrides, args.test_seeds, args.total_steps)

    print(f"\n{'Config':<10} | {'Split':<12} | {'Mean return':>12} | {'Std':>8}")
    print(f"{'BO':<10} | {'opt-seeds':<12} | {np.mean(bo_opt_scores):12.2f} | {np.std(bo_opt_scores):8.2f}")
    print(f"{'BO':<10} | {'test-seeds':<12} | {np.mean(bo_test_scores):12.2f} | {np.std(bo_test_scores):8.2f}")
    print(f"{'Default':<10} | {'opt-seeds':<12} | {np.mean(base_opt_scores):12.2f} | {np.std(base_opt_scores):8.2f}")
    print(f"{'Default':<10} | {'test-seeds':<12} | {np.mean(base_test_scores):12.2f} | {np.std(base_test_scores):8.2f}")


if __name__ == "__main__":
    main()
