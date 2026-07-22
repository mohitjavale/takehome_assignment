import os
from omegaconf import OmegaConf

from humanoid_standup.envs.humanoid_standup_env import HumanoidStandupEnv
import torch

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
cfg = OmegaConf.load(config_path)

cfg.env.num_envs = 10

env = HumanoidStandupEnv(cfg, device="cuda", show_viewer=True)
obs = env.reset()

while True:
    zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, rew, reset, extras = env.step(zero_actions)
        

