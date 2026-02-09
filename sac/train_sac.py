from pettingzoo.sisl import multiwalker_v9
import supersuit as ss

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from train_environment import make_env


train_env = make_env(n_walkers=3)
eval_env = make_env(n_walkers=3)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./sac_multiwalker/best/",
    log_path="./sac_multiwalker/logs/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

model = SAC(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./sac_multiwalker_tensorboard/",
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef='auto',
    policy_kwargs=dict(net_arch=[256, 256]),
)


model.learn(
    total_timesteps=2_000_000,
    callback=eval_callback,
    progress_bar=True
)


model.save("./sac_multiwalker/final_model")
