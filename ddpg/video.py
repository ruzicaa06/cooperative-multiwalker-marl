from stable_baselines3 import DDPG
from pettingzoo.sisl import multiwalker_v9
from moviepy import ImageSequenceClip
import supersuit as ss
import numpy as np

render_env = multiwalker_v9.parallel_env(
    n_walkers=3,
    render_mode="rgb_array"
)

model = DDPG.load("./ddpg_multiwalker/best/best_model")

frames = []

obs, infos = render_env.reset()
while render_env.agents:
    frame = render_env.render()
    frames.append(frame)

    actions = {}
    for agent, agent_obs in obs.items():
        action, _ = model.predict(
            agent_obs,
            deterministic=True
        )
        actions[agent] = action

    obs, rewards, terminations, truncations, infos = render_env.step(actions)

render_env.close()

clip = ImageSequenceClip(frames, fps=30)
clip.write_videofile(
    "./ddpg_video.mp4",
    codec="libx264",
    audio=False
)
