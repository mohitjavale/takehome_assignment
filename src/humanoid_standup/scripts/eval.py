import os
import sys
import glob

from humanoid_standup.envs.humanoid_standup_env import HumanoidStandupEnv
from humanoid_standup.models.actor_critic import Policy, Value

from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

from omegaconf import OmegaConf

def get_latest_checkpoint(base_dir="runs", exp_name="g1_standup"):
    search_pattern = os.path.join(base_dir, f"{exp_name}*")
    run_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    
    
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_run_dir = run_dirs[0]
    
    checkpoints_dir = os.path.join(latest_run_dir, "checkpoints")         
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith("agent_") and f.endswith(".pt")]
        
    def get_step(filename):
        try:
            return int(filename.split('_')[1].split('.')[0])
        except ValueError:
            return -1
            
    checkpoints.sort(key=get_step, reverse=True)
    return os.path.join(checkpoints_dir, checkpoints[0])

def main():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    checkpoint_path = cfg.get("checkpoint", None)
    if not checkpoint_path:
        checkpoint_path = get_latest_checkpoint(base_dir="runs", exp_name=cfg.training.exp_name)
            
    print(f"Loading checkpoint from: {checkpoint_path}")

    cfg.env.num_envs = 16  
    
    env = HumanoidStandupEnv(cfg, device="cpu", show_viewer=True)
    env = wrap_env(env, wrapper="isaacgym-preview4")
    device = env.device

    models = {
        "policy": Policy(env.observation_space, env.action_space, device, cfg.models.policy, clip_actions=False),
        "value": Value(env.observation_space, env.action_space, device, cfg.models.value)
    }

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    agent = PPO(models=models,
                memory=None,
                cfg=ppo_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    agent.load(checkpoint_path)
    agent.set_mode("eval")  

    obs, info = env.reset()

    print("\n--- Starting evaluation loop. Press Ctrl+C to stop. ---\n")    
    try:
        while True:
            actions = agent.act(obs, timestep=0, timesteps=0)[0]
            obs, reward, terminated, truncated, info = env.step(actions)
            
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")

if __name__ == "__main__":
    main()