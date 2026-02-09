from pettingzoo.sisl import multiwalker_v9
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch as th

from train_environment import make_env

train_env = make_env(n_walkers=3)
eval_env = make_env(n_walkers=3)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="ppo_multiwalker_opt/best/",
    log_path="ppo_multiwalker_opt/logs/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./ppo_multiwalker_tensorboard_opt/",
    learning_rate=0.00016266495152941822,
    n_steps=1024,
    batch_size=32,
    n_epochs=15,
    gamma=0.9857372136462389,
    gae_lambda=0.9449479763419787,
    clip_range=0.3961323822082683,
    ent_coef=0.019576816559393886,
    max_grad_norm=0.5701680875305213,
    vf_coef=0.8757372136462389,
    policy_kwargs=dict(
        net_arch=[512, 512]
    )
)

model.learn(
    total_timesteps=2_000_000,
    callback=eval_callback,
    progress_bar=True
)


model.save("./ppo_multiwalker_opt/final_model")
