from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from pettingzoo.sisl import multiwalker_v9

import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt

def make_env(n_walkers=3):
    env = multiwalker_v9.parallel_env(n_walkers=n_walkers)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env

env = make_env()
model = DDPG.load("./ddpg_multiwalker/best/best_model.zip", env=env)

# mean and std
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=50,
    deterministic=True
)

print(f"DDPG Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

data = np.load("./ddpg_multiwalker/logs/evaluations.npz")


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
plt.title("DDPG performance on MultiWalker")
plt.grid(True)

plt.savefig("./ddpg_learning_curve.pdf", bbox_inches="tight")
plt.show()
