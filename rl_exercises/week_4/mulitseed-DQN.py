import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rliable import library as rly
from rliable import metrics, plot_utils


def train_seed(seed: int, num_frames: int = 50000):
    env = gym.make("CartPole-v1")
    set_seed(env, seed)

    agent = DQNAgent(
        env,
        buffer_capacity=10000,
        batch_size=32,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500,
        target_update_freq=1000,
        seed=seed,
        hidden_dim=64,
        depth=2,
    )

    frames, rewards = agent.train(num_frames=num_frames)
    env.close()

    return np.array(frames), np.array(rewards)


def aggregate_func(x):
    """
    Aggregates metric scores by calculating the median, interquartile mean (IQM),
    mean, and optimality gap.

    Args:
        x: The array of scores to aggregate.

    Returns:
        A NumPy array containing the aggregated metrics.
    """
    return np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x, gamma=500),
        ]
    )


def main():
    seeds = [0, 1, 2, 3, 4]

    all_rewards = []

    for seed in seeds:
        print(f"Training seed {seed}...")
        _, rewards = train_seed(seed)
        all_rewards.append(rewards)

    min_len = min(len(r) for r in all_rewards)
    rewards = np.array([r[:min_len] for r in all_rewards])

    final_scores = rewards[:, -1][:, None]

    score_dict = {"DQN": final_scores}

    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=50000
    )

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        figsize=(16, 6),
    )

    try:
        for ax in axes.ravel():
            ax.set_ylabel("")
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontsize(10)
    except Exception:
        pass

    for text_obj in fig.texts:
        if "Normalized Score" in text_obj.get_text():
            text_obj.set_visible(False)

    plt.savefig("mulitseed-DQN.png", dpi=300, bbox_inches="tight")

    print("Saved: mulitseed-DQN.png")


if __name__ == "__main__":
    main()
