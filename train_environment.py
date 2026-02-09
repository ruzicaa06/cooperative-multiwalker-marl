from pettingzoo.sisl import multiwalker_v9
import supersuit as ss

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