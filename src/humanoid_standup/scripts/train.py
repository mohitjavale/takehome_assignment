import os

from humanoid_standup.envs.humanoid_standup_env import HumanoidStandupEnv
from humanoid_standup.models.actor_critic import Policy, Value

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveLR

from omegaconf import OmegaConf

def main():
    # config stuff
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    run_dir = os.path.join("runs", cfg.training.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    # create env
    env = HumanoidStandupEnv(cfg, device="cuda:0", show_viewer=False)
    env = wrap_env(env, wrapper="isaacgym-preview4")
    device = env.device

    # policy stuff
    models = {
        "policy": Policy(env.observation_space, env.action_space, device, cfg.models.policy, clip_actions=False),
        "value": Value(env.observation_space, env.action_space, device, cfg.models.value)
    }

    rollouts = cfg.training.rollouts
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["rollouts"] = rollouts
    ppo_cfg["learning_epochs"] = 5 # 16         
    ppo_cfg["mini_batches"] = 4 # 16            
    ppo_cfg["discount_factor"] = cfg.training.discount_factor
    ppo_cfg["lambda"] = cfg.training["lambda"]
    ppo_cfg["learning_rate"] = cfg.training.learning_rate
    ppo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
    ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": cfg.training.kl_threshold}
    ppo_cfg["entropy_loss_scale"] = 0.01

    ppo_cfg["experiment"]["directory"] = "runs"
    ppo_cfg["experiment"]["experiment_name"] = cfg.training.exp_name
    ppo_cfg["experiment"]["write_interval"] = "auto"
    ppo_cfg["experiment"]["checkpoint_interval"] = cfg.training.save_interval * rollouts
    ppo_cfg["experiment"]["store_separately"] = False
    ppo_cfg["experiment"]["wandb"] = False

    agent = PPO(models=models,
                memory=memory,
                cfg=ppo_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
    
    _orig_post_interaction = agent.post_interaction
    def post_interaction(timestep, timesteps):
        for k, v in env._env.extras.items():
            if k.startswith("Reward / "):
                agent.track_data(k, v.item() if hasattr(v, "item") else v)
        _orig_post_interaction(timestep, timesteps)
    agent.post_interaction = post_interaction

    cfg_trainer = {
        "timesteps": cfg.training.max_iterations * rollouts, 
        "headless": True # we are mantaining own rendering ppipeline
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()

if __name__ == "__main__":
    main()