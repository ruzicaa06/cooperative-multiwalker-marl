from pettingzoo.sisl import multiwalker_v9
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch as th

def make_env(n_walkers=3):
    env = multiwalker_v9.parallel_env(
        n_walkers=n_walkers,
        position_noise=1e-3,
        angle_noise=1e-3
    )

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3"
    )
    return env

def lr_schedule(progress):
    return max(1e-4, progress * 3e-4)


train_env = make_env(n_walkers=3)
eval_env = make_env(n_walkers=3)


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./ppo_multiwalker/best/",
    log_path="./ppo_multiwalker/logs/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./ppo_multiwalker_tensorboard/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
)

model.learn(
    total_timesteps=2_000_000,
    callback=eval_callback,
    progress_bar=True
)


model.save("./ppo_multiwalker/final_model")
