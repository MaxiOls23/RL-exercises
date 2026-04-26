import numpy as np
from rl_exercises.environments import MarsRover
from rl_exercises.train_agent import evaluate
from rl_exercises.week_2 import PolicyIteration
from rl_exercises.week_2.contextual_mars_rover import ContextualMarsRover


def evaluate_context_aware_unaware():
    seed = 1
    episodes = 100

    train_set = [
        {"slip": 0.0, "goal_reward": 10.0},
        {"slip": 0.3, "goal_reward": 15.0},
        {"slip": 0.4, "goal_reward": 9.0},
    ]
    val_set = [{"slip": 0.1, "goal_reward": 8.0}]
    test_set = [{"slip": 0.8, "goal_reward": 1.5}]

    env_contextual = ContextualMarsRover(seed=seed)
    n_s, n_a = env_contextual.observation_space.n, env_contextual.action_space.n

    avg_T = np.zeros((n_s, n_a, n_s))
    avg_R_sa = np.zeros((n_s, n_a))

    for ctx in train_set:
        env_contextual.set_context(ctx)
        avg_T += env_contextual.transition_matrix
        avg_R_sa += env_contextual.get_reward_per_action()

    avg_T /= len(train_set)
    avg_R_sa /= len(train_set)

    env_normal = MarsRover(seed=seed)

    def evaluate_performance(contexts, label):
        print(f"\n--- {label} Set Evaluation ---")
        print(f"{'Context':<40} | {'Unaware V':<12} | {'Aware V':<12}")
        print("-" * 75)

        for ctx in contexts:
            env_contextual.set_context(ctx)

            v_unaware = PolicyIteration(env=env_normal, seed=seed)
            v_unaware.T = avg_T
            v_unaware.R_sa = avg_R_sa
            v_unaware.update_agent()
            mean_unaware = evaluate(
                env=env_contextual, agent=v_unaware, episodes=episodes
            )

            v_aware = PolicyIteration(env=env_contextual, seed=seed)
            v_aware.update_agent()
            mean_aware = evaluate(env=env_contextual, agent=v_aware, episodes=episodes)
            print(f"{str(ctx):<40} | {mean_unaware:<12.4f} | {mean_aware:<12.4f}")

    evaluate_performance(train_set, "Train")
    evaluate_performance(val_set, "Validation")
    evaluate_performance(test_set, "Test (OOD)")


if __name__ == "__main__":
    evaluate_context_aware_unaware()
