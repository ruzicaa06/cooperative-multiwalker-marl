import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import supersuit as ss
from pettingzoo.sisl import multiwalker_v9
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
import numpy as np
import os


def make_env(n_walkers=3, seed=None):
    env = multiwalker_v9.parallel_env(
        n_walkers=n_walkers,
        position_noise=1e-3,
        angle_noise=1e-3,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3"
    )
    if seed is not None:
        env.seed(seed)
    return env


def objective(trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    gamma = trial.suggest_float("gamma", 0.95, 0.999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
    vf_coef = trial.suggest_float("vf_coef", 0.2, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    net_arch_size = trial.suggest_categorical("net_arch_size", [128, 256, 512])
    net_arch = [net_arch_size, net_arch_size]

    train_env = make_env(n_walkers=3)
    eval_env = make_env(n_walkers=3)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"../optuna_ppo/optuna_ppo_trials/trial_{trial.number}/best_model",
        log_path=f"../optuna_ppo/optuna_ppo_trials/trial_{trial.number}/logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0,
        tensorboard_log=None,
        device="cuda" if th.cuda.is_available() else "cpu",
    )


    TIMESTEPS_FOR_TRIAL = 300_000
    model.learn(
        total_timesteps=TIMESTEPS_FOR_TRIAL,
        callback=eval_callback,
        progress_bar=False,
    )

    best_mean_reward = eval_callback.best_mean_reward

    if best_mean_reward == -np.inf:
        mean_reward, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
            deterministic=True,
        )
        best_mean_reward = mean_reward


    train_env.close()
    eval_env.close()

    return best_mean_reward


if __name__ == "__main__":
    os.makedirs("optuna_ppo_trials", exist_ok=True)

    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="ppo_multiwalker_optuna",
        storage="sqlite:///ppo_multiwalker_optuna.db",  # persistent
        load_if_exists=True,
    )

    print("Starting hyperparameter optimization...")
    study.optimize(
        objective,
        n_trials=60,
        timeout=3600 * 8,
        show_progress_bar=True,
        n_jobs=1,
    )

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.3f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    import json
    with open("best_ppo_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
