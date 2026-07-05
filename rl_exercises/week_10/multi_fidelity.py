from __future__ import annotations

from typing import Any, Dict, List

import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_exercises.week_9.dyna_ppo import DynaPPOAgent

# Search space for hyperparameters
SEARCH_SPACE = {
    "imag_horizon": (1, 4),
    "imag_batches": (1, 5),
    "model_epochs": (1, 2),
}

SCORE_ALPHA = 0.1

# Default agent configuration
DEFAULT_CONFIG = dict(
    epochs=4,
    batch_size=64,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    hidden_size=64,
    use_model=True,
    model_lr=1e-3,
    model_epochs=3,
    model_batch_size=64,
    imag_horizon=5,
    imag_batches=10,
    max_buffer_size=10000,
)


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
    eval_interval: int = 500,
    eval_episodes: int = 5,
    seed: int = 42,
) -> Dict[str, List]:
    eval_env = gym.make(agent.env.spec.id)
    eval_env.reset(seed=seed + 1)

    metrics: Dict[str, List] = {
        "real_steps": [],
        "returns": [],
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
                (
                    state,
                    action,
                    logp_t.squeeze(),
                    ent_t.squeeze(),
                    float(reward),
                    float(done),
                    next_state,
                )
            )
            state = next_state
            agent.real_steps += 1

            if agent.real_steps % eval_interval == 0:
                mean_r, _ = agent.evaluate(eval_env, num_episodes=eval_episodes)
                metrics["real_steps"].append(agent.real_steps)
                metrics["returns"].append(mean_r)

        agent.total_episodes += 1
        agent.update(real_traj)

        if agent.use_model:
            agent.store_real(real_traj)
            agent.train_model()
            agent.imagine_and_update()

    return metrics


def evaluate_agents_to_budget(
    agent_list: List[DynaPPOAgent],
    budget_steps: int,
    seeds: List[int],
) -> float:
    """Advance existing agents to a specific step budget and compute performance score."""
    finals = []
    for agent, seed in zip(agent_list, seeds):
        metrics = train_and_collect(
            agent, total_steps=budget_steps, eval_interval=500, seed=seed
        )
        finals.append(
            float(np.mean(metrics["returns"][-2:])) if metrics["returns"] else 0.0
        )

    return float(np.mean(finals) - SCORE_ALPHA * np.std(finals))


def run_level2_experiment() -> None:
    opt_seeds = [1, 2]
    rng = np.random.default_rng(42)
    names = list(SEARCH_SPACE.keys())
    bounds = np.array([SEARCH_SPACE[n] for n in names], dtype=float)

    configs = []
    for _ in range(8):
        x = rng.integers(bounds[:, 0], bounds[:, 1] + 1).astype(float)
        configs.append({n: int(round(v)) for n, v in zip(names, x)})

    # The targeted setup that suffers from poor initial dynamics but thrives on higher allocations.
    late_bloomer = {"imag_horizon": 12, "imag_batches": 12, "model_epochs": 5}
    configs.append(late_bloomer)

    pool_state = []
    for cfg in configs:
        agents = [make_agent(seed=s, **cfg) for s in opt_seeds]
        pool_state.append({"config": cfg, "agents": agents})

    budgets = [500, 1500, 4500, 10000]
    eliminated_late_bloomer_early = False

    for r, budget in enumerate(budgets):
        print(f"\n--- Round {r + 1} (Fidelity Budget: {budget} Steps) ---")
        print(f"Evaluating {len(pool_state)} remaining configurations...")

        round_results = []
        for i, entry in enumerate(pool_state):
            cfg = entry["config"]
            score = evaluate_agents_to_budget(entry["agents"], budget, opt_seeds)
            is_bloomer = cfg == late_bloomer
            label = " [Late Bloomer Target]" if is_bloomer else ""
            print(f"  Config {i + 1}: {cfg} -> Score: {score:.2f}{label}")
            round_results.append((score, entry))

        round_results.sort(key=lambda x: x[0], reverse=True)

        # Enforce structural compression via elimination steps while guarding boundaries.
        next_size = max(1, len(pool_state) // 3)
        surviving_results = round_results[:next_size]
        survivors = [entry for _, entry in surviving_results]

        remaining_configs = [e["config"] for e in survivors]
        if (
            late_bloomer in [e["config"] for e in pool_state]
            and late_bloomer not in remaining_configs
        ):
            if r == 0:
                eliminated_late_bloomer_early = True
                print(
                    "\nANOMALY DETECTED: Late bloomer eliminated at early budget step (500 steps)!"
                )

        pool_state = survivors
        print(f"{len(pool_state)} configurations advance to the next stage.")

    print(
        "\nTesting the late bloomer setup alongside winner on full budget (10000 steps)"
    )

    winner_cfg = pool_state[0]["config"]
    winner_full_score = round_results[0][0]

    # Evaluate the late bloomer separately from scratch to obtain its performance capacity at terminal capacity.
    fresh_bloomer_agents = [make_agent(seed=s, **late_bloomer) for s in opt_seeds]
    bloomer_full_score = evaluate_agents_to_budget(
        fresh_bloomer_agents, 10000, opt_seeds
    )

    print("\nResult at full fidelity budget (10000 steps):")
    print(
        f"Config selected by SHA optimizer: {winner_cfg} -> Score: {winner_full_score:.2f}"
    )
    print(
        f"Config eliminated too early: {late_bloomer} -> Score: {bloomer_full_score:.2f}"
    )

    write_results_txt(
        winner_cfg,
        winner_full_score,
        late_bloomer,
        bloomer_full_score,
        eliminated_late_bloomer_early,
    )


def write_results_txt(winner, winner_score, bloomer, bloomer_score, early_drop):
    with open("level2_multi_fidelity.txt", "w") as f:
        f.write("Level 2 Multi-fidelity RL Results\n\n")
        f.write("1. Scenario description:\n")
        f.write(
            "In model-based RL like Dyna-PPO, early performance can be deceptive.\n"
        )
        f.write(
            "At low fidelity budgets (500 steps), the world model is untrained and inaccurate.\n"
        )
        f.write(
            "Setups with high imag_horizon use bad model data early on and get low scores,\n"
        )
        f.write("making them look bad to multi-fidelity pruners.\n\n")

        f.write("2. Optimization history and analysis:\n")
        if early_drop:
            f.write(
                "The late bloomer configuration was eliminated in round 1 (budget 500 steps).\n"
            )
            f.write("The early short-term signals tricked the optimizer.\n")

        f.write(
            f"Optimizer selected configuration: {winner} (Score: {winner_score:.2f})\n"
        )
        f.write(
            f"Early eliminated late bloomer:    {bloomer} (Score: {bloomer_score:.2f})\n\n"
        )

        f.write("3. Conclusion:\n")
        if bloomer_score > winner_score:
            f.write("Yes, multi-fidelity optimization was negatively affected.\n")
            f.write("An excellent final hyperparameter setup was dropped too early\n")
            f.write("because early metrics did not correlate with final performance.\n")


if __name__ == "__main__":
    run_level2_experiment()
