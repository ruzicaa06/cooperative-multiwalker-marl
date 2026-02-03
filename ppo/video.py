from stable_baselines3 import PPO
from pettingzoo.sisl import multiwalker_v9
from moviepy import ImageSequenceClip
import supersuit as ss
import numpy as np

render_env = multiwalker_v9.parallel_env(
    n_walkers=3,
    render_mode="rgb_array"
)

model = PPO.load("./ppo_multiwalker/best/best_model.zip")

frames = []

obs, infos = render_env.reset()
while render_env.agents:
    frame = render_env.render()
    frames.append(frame)

    actions = {}
    for agent, agent_obs in obs.items():
        action, _ = model.predict(
            agent_obs,
            deterministic=False
        )
        actions[agent] = action

    obs, rewards, terminations, truncations, infos = render_env.step(actions)

render_env.close()

clip = ImageSequenceClip(frames, fps=30)
clip.write_videofile(
    "./ppo_video_non_dem_2.mp4",
    codec="libx264",
    audio=False
)
