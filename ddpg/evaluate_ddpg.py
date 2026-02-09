from stable_baselines3 import DDPG
from pettingzoo.sisl import multiwalker_v9

import supersuit as ss
import numpy as np

from evaluate import mean_and_std, plot_learning_curve
from film_video import film

def make_env(n_walkers=3):
    env = multiwalker_v9.parallel_env(n_walkers=n_walkers)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env

env = make_env()
model = DDPG.load("./ddpg_multiwalker/best/best_model.zip", env=env)
model_name ="DDPG"

# mean and std
mean, std = mean_and_std(model_name, model, env)

# learning curve
data = np.load("./ddpg_multiwalker/logs/evaluations.npz")
plot_learning_curve(model_name, data)

# video
film(model_name, model)
