import os
import sys
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch
import gym

class HumanoidStandupEnv(gym.Env):
    def __init__(self, cfg, device="cuda:0", show_viewer=False):
        super().__init__()

        # skrl required stuff
        self.device = torch.device(device)
        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions
        self.num_obs = cfg.obs.num_obs
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.num_obs,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        self.cfg = cfg
        self.show_viewer = show_viewer
        self.decimation = self.cfg.env.control.decimation
        self.sim_dt = cfg.env.sim.dt
        self.dt = self.sim_dt * self.decimation        
        self.max_episode_length = int(cfg.env.episode_length_s / self.dt)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat((self.num_envs, 1))

        self.gym = gymapi.acquire_gym()
        self._create_sim()
        self._create_ground_plane()
        self._load_asset()
        self._create_envs()
        self._init_buffers()
        self._create_viewer()

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.substeps = self.cfg.env.sim.substeps
        sim_params.dt = self.cfg.env.sim.dt
        sim_params.use_gpu_pipeline = self.device.type == "cuda"
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(*self.cfg.env.sim.gravity)
        
        if self.device.type == "cuda":
            sim_params.physx.use_gpu = True
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.contact_offset = 0.02
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.bounce_threshold_velocity = 0.2
            sim_params.physx.max_depenetration_velocity = 10.0
            sim_params.physx.default_buffer_size_multiplier = 5.0
            
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _load_asset(self):
        asset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", self.cfg.env.asset.folder))
        asset_file = self.cfg.env.asset.file
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(self.asset)
        self.num_rb = self.gym.get_asset_rigid_body_count(self.asset)

        kp = self.cfg.env.control.kp
        kd = self.cfg.env.control.kd
        p_gains, d_gains = [], []

        for dof_name in self.cfg.env.asset.dof_names:
            matched_key = next((k for k in kp.keys() if k in dof_name), None)
            p_gains.append(kp[matched_key])
            d_gains.append(kd[matched_key])

        self.p_gains = torch.tensor(p_gains, dtype=torch.float32, device=self.device, requires_grad=False)
        self.d_gains = torch.tensor(d_gains, dtype=torch.float32, device=self.device, requires_grad=False)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        if self.cfg.env.control.use_sim_PD_controller:
            self.dof_props["driveMode"] = gymapi.DOF_MODE_POS
            self.dof_props['stiffness'] = self.p_gains.cpu().numpy()
            self.dof_props['damping'] = self.d_gains.cpu().numpy()
        else:
            self.dof_props["driveMode"] = gymapi.DOF_MODE_EFFORT
            self.dof_props['stiffness'].fill(0.0)
            self.dof_props['damping'].fill(0.0)

        foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_rb_indices = [self.gym.find_asset_rigid_body_index(self.asset, name) for name in foot_link_names]

    def _create_envs(self):
        spacing = self.cfg.env.spacing
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actor_handles = []
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.cfg.env.asset.base_init_pos)
        pose.r = gymapi.Quat(*self.cfg.env.asset.base_init_quat)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, 10)
            handle = self.gym.create_actor(env, self.asset, pose, "robot", i, 1)
            self.gym.set_actor_dof_properties(env, handle, self.dof_props)
            self.envs.append(env)
            self.actor_handles.append(handle)

        self.gym.prepare_sim(self.sim)

    def _create_viewer(self):
        self.viewer = None
        self.enable_viewer_sync = True
        if self.show_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            cam_pos = gymapi.Vec3(2.0, 2.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _init_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_states)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states).view(self.num_envs, self.num_dof, 2)
        self.dof_pos = self.dof_states[:, :, 0]
        self.dof_vel = self.dof_states[:, :, 1]
        
        default_angles = [self.cfg.env.asset.default_joint_angles[name] for name in self.cfg.env.asset.dof_names]
        self.default_dof_pos = torch.tensor(default_angles, dtype=torch.float32, device=self.device, requires_grad=False)

        self.dof_lower = torch.tensor(self.dof_props['lower'], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_upper = torch.tensor(self.dof_props['upper'], dtype=torch.float32, device=self.device, requires_grad=False)

        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(self.num_envs, self.num_rb, 3)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_root_states = self.root_states.clone()

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return {"obs": self.obs_buf}
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def step(self, actions):
        self.last_actions[:] = self.actions[:]
        self.actions = torch.clip(actions.detach(), -self.cfg.env.clip_actions, self.cfg.env.clip_actions)
        
        if self.cfg.env.control.use_sim_PD_controller:
            targets = (self.actions * self.cfg.env.action_scale) + self.default_dof_pos
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        
        for _ in range(self.decimation):
            if not self.cfg.env.control.use_sim_PD_controller:
                self.gym.refresh_dof_state_tensor(self.sim)
                torques = self.batched_p_gains * ((self.actions * self.cfg.env.action_scale + self.default_dof_pos) - self.dof_pos) \
                          - self.batched_d_gains * self.dof_vel
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            
            self.gym.simulate(self.sim)
            
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.show_viewer:
            self.render()
                
        self.episode_length_buf += 1

        self.compute_observations()
        self.compute_rewards()
        self.check_termination()

        self.extras["truncated"] = self.reset_buf.clone()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        return {"obs": self.obs_buf}, self.rew_buf, self.reset_buf, self.extras
    
    def _random_quaternions(self, n):
        u1 = torch.rand(n, device=self.device)
        u2 = torch.rand(n, device=self.device) * 2 * torch.pi
        u3 = torch.rand(n, device=self.device) * 2 * torch.pi
        x = torch.sqrt(1 - u1) * torch.sin(u2)
        y = torch.sqrt(1 - u1) * torch.cos(u2)
        z = torch.sqrt(u1) * torch.sin(u3)
        w = torch.sqrt(u1) * torch.cos(u3)
        return torch.stack([x, y, z, w], dim=-1)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        n = len(env_ids)

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 3:7] = self._random_quaternions(n)
        self.root_states[env_ids, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids))
        
        rand_frac = torch.rand((n, self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.dof_lower + rand_frac * (self.dof_upper - self.dof_lower)
        self.dof_vel[env_ids] = 0.0
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids))

        self.last_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False 

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.compute_observations()

    def compute_observations(self):
        base_quat = self.root_states[:, 3:7]
        base_lin_vel = self.root_states[:, 7:10]
        base_ang_vel = self.root_states[:, 10:13]

        base_lin_vel_local = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel_local = quat_rotate_inverse(base_quat, base_ang_vel)
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        scales = self.cfg.obs.scales
        self.obs_buf = torch.cat((
            base_lin_vel_local * scales.lin_vel,
            base_ang_vel_local * scales.ang_vel,
            projected_gravity,
            (self.dof_pos - self.default_dof_pos) * scales.dof_pos,
            self.dof_vel * scales.dof_vel,
            self.actions
        ), dim=-1)


    def compute_rewards(self):
        self.rew_buf[:] = 0.0
        self.extras.clear()
        
        scales = self.cfg.reward.scales


        base_height = self.root_states[:, 2]
        height_frac = torch.clamp(base_height / scales.target_height, min=0.0, max=1.0)

        base_quat = self.root_states[:, 3:7]
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        upright_frac = torch.clamp(-projected_gravity[:, 2], min=0.0, max=1.0)

        standup_reward = height_frac * upright_frac * scales.standup

        dof_pos_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        pose_reward = torch.exp(-dof_pos_error / scales.pose_sigma) * height_frac * scales.pose

        action_rate_penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * scales.action_rate
        dof_vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1) * scales.dof_vel

        self.rew_buf += standup_reward + pose_reward + action_rate_penalty + dof_vel_penalty

        #  Log to extras to be picked by tensorboard writer for logging
        self.extras["Reward / standup_reward"] = standup_reward.mean()
        self.extras["Reward / pose_reward"] = pose_reward.mean()
        self.extras["Reward / action_rate_penalty"] = action_rate_penalty.mean()
        self.extras["Reward / dof_vel_penalty"] = dof_vel_penalty.mean()
        is_standing = (height_frac > 0.8) & (upright_frac > 0.8)
        self.extras["is_standing"] = is_standing.clone()


    # def compute_rewards(self):
    #     self.rew_buf[:] = 0.0
    #     self.extras.clear()
        
    #     scales = self.cfg.reward.scales


    #     base_height = self.root_states[:, 2]
    #     height_frac = torch.clamp(base_height / scales.target_height, min=0.0, max=1.0)

    #     base_quat = self.root_states[:, 3:7]
    #     projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
    #     upright_frac = torch.clamp(-projected_gravity[:, 2], min=0.0, max=1.0)

    #     standup_reward = height_frac * upright_frac * scales.standup

    #     dof_pos_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    #     pose_reward = torch.exp(-dof_pos_error / scales.pose_sigma) * height_frac * scales.pose

    #     action_rate_penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * scales.action_rate
    #     dof_vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1) * scales.dof_vel

    #     is_high = (height_frac > 0.5).float()
    #     base_z_vel_penalty = torch.square(self.root_states[:, 9]) * scales.base_z_vel * is_high
    #     base_ang_vel_penalty = torch.sum(torch.square(self.root_states[:, 10:13]), dim=1) * scales.base_ang_vel


    #     self.rew_buf += (standup_reward + pose_reward + action_rate_penalty + 
    #                      dof_vel_penalty + base_z_vel_penalty + base_ang_vel_penalty)

    #     self.extras["Reward / standup_reward"] = standup_reward.mean()
    #     self.extras["Reward / pose_reward"] = pose_reward.mean()
    #     self.extras["Reward / action_rate_penalty"] = action_rate_penalty.mean()
    #     self.extras["Reward / dof_vel_penalty"] = dof_vel_penalty.mean()
    #     self.extras["Reward / base_z_vel_penalty"] = base_z_vel_penalty.mean()
    #     self.extras["Reward / base_ang_vel_penalty"] = base_ang_vel_penalty.mean()

    #     is_standing = (height_frac > 0.8) & (upright_frac > 0.8)
    #     self.extras["is_standing"] = is_standing.clone()



    def check_termination(self):
        self.reset_buf = self.episode_length_buf >= self.max_episode_length



