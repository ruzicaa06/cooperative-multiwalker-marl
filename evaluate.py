from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt


def mean_and_std(model_name, model, env, n_episodes=100):
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True
    )

    print(f"{model_name} Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward


def plot_learning_curve(model_name, data):
    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.figure()
    plt.plot(timesteps, mean_rewards)
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3
    )

    plt.xlabel("Training timesteps")
    plt.ylabel("Mean evaluation reward")
    plt.title(f"{model_name} performance on MultiWalker")
    plt.grid(True)

    plt.savefig(f"./{model_name}_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.show()