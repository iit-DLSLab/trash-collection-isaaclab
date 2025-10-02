# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns, Imu
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


from .arm_env_cfg import ArmFlatEnvCfg, ArmRoughBlindEnvCfg, ArmRoughVisionEnvCfg

from trash_collection_isaaclab.tasks.supervised_learning_networks import SimpleNN

class ManipulationEnv(DirectRLEnv):
    cfg: ArmFlatEnvCfg | ArmRoughBlindEnvCfg | ArmRoughVisionEnvCfg

    def __init__(self, cfg: ArmFlatEnvCfg | ArmRoughBlindEnvCfg | ArmRoughVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        
        # pose ee commands #TODO add roll pitch yaw
        self._ee_commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Observation history arm
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)

        # Observation history locomotion
        self._observation_history_locomotion = torch.zeros(self.num_envs, cfg.history_length, cfg.single_locomotion_observation_space, device=self.device)
        if self.cfg.observation_noise_model:
            self._observation_noise_model_locomotion: NoiseModel = self.cfg.observation_noise_model.class_type(
                self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )

        # RMA
        if(cfg.use_rma == True):
            self._rma_network = SimpleNN(cfg.rma_observation_space, cfg.rma_output_space)
            self._rma_network.to(self.device)
            self._observation_history_rma = torch.zeros(self.num_envs, cfg.history_length, cfg.single_rma_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )

        # Periodic gait
        if(cfg.desired_gait == "trot"):
            self._step_freq = 1.4
            self._duty_factor = 0.65
            self._phase_offset = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).repeat(self.num_envs,1)
            self._velocity_gait_multiplier = 1.0
        elif(cfg.desired_gait == "crawl"):
            self._step_freq = 0.5
            self._duty_factor = 0.8
            self._phase_offset = torch.tensor([0.0, 0.5, 0.75, 0.25], device=self.device).repeat(self.num_envs,1)
            self._velocity_gait_multiplier = 0.5
        elif(cfg.desired_gait == "pace"):
            self._step_freq = 1.4
            self._duty_factor = 0.7
            self._phase_offset = torch.tensor([0.8, 0.3, 0.8, 0.3], device=self.device).repeat(self.num_envs,1)
            self._velocity_gait_multiplier = 1.0
        elif(cfg.desired_gait == "multigait"):
            #TODO: implement multigait
            raise NotImplementedError("Multigait not implemented yet")
        self._phase_signal = self._phase_offset.clone()# + self.step_dt * self._step_freq * torch.rand(self.num_envs, 1, device=self.device)*10.
        self._phase_signal = self._phase_signal % 1.0


        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_ee_exp",

                "undesired_contacts",
                "action_rate_l2",
                "action_smoothness_l2",
                
                "joints_arm_l2",
                "joints_acc_l2",
                "joints_torques_l2",
                "joints_energy_l1",
                
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._undesired_contact_body_ids = self._base_id + self._hip_ids + self._thigh_ids

        
        self._feet_ids_robot, _ = self._robot .find_bodies(".*foot")
        self._hip_ids_robot, _ = self._robot.find_bodies(".*hip")
        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]
        self._ids_only_legs_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order[0:12], preserve_order=True)[0]
        self._ids_only_arms_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order[12:18], preserve_order=True)[0]


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # we add an imu
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        
        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order]
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order]


    def _apply_action(self):
        processed_actions_with_arm = torch.zeros(self.num_envs, 18, device=self.device)
        processed_actions_with_arm[:, self._ids_only_arms_joints_order] = self._processed_actions

        # Get locomotion policy action
        locomotion_actions = self._get_locomotion_policy_action()
        processed_actions_with_arm[:, self._ids_only_legs_joints_order] = locomotion_actions
        self._robot.set_joint_position_target(processed_actions_with_arm)


    def _get_observations(self) -> dict:
        
        # This is a custom event, to be moved in custom_events.py
        self._get_new_random_commands()


        # Observation --------------------------------------------------------------------------------------
        # Standard Obs for the Actor/Critic
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._ee_commands,
                    self._robot.data.joint_pos[:,self._ids_only_arms_joints_order] - self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order],
                    self._robot.data.joint_vel[:,self._ids_only_arms_joints_order],
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)


        # Add heightmap data to obs if needed
        if isinstance(self.cfg, ArmRoughVisionEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)      


        # If RMA, we add some other predicted obs
        if(self.cfg.use_rma):
            # Predict the RMA observation
            obs_rma = self._get_rma(None)
            obs = torch.cat((obs, obs_rma), dim=-1)


        # Final observations dictionary
        observations = {"policy": obs}    
        

        # Critic OBS could be different if needed
        if(self.cfg.use_asymmetric_ppo):
            obs_critic = self._get_privileged_observation()
            observations["critic"] = torch.cat((obs, obs_critic), dim=-1)
        # ------------------------------------------------------------------------------------------
        return observations


    def _get_rewards(self) -> torch.Tensor:
        
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_smoothness = torch.sum(torch.square(self._actions - 2*self._previous_actions + self._previous_previous_actions), dim=1)
        
        
        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        

        # joint acceleration
        joints_accel = torch.sum(torch.square(self._robot.data.joint_acc[:,self._ids_only_arms_joints_order]), dim=1)


        # joint torques
        joints_torques = torch.sum(torch.square(self._robot.data.applied_torque[:,self._ids_only_arms_joints_order]), dim=1)


        # energy = torque * velocity
        joints_energy = torch.sum(torch.abs(self._robot.data.applied_torque[:,self._ids_only_arms_joints_order] * self._robot.data.joint_vel[:,self._ids_only_arms_joints_order]), dim=1)

        
        # joints position
        joints_arm_position = self._robot.data.joint_pos[:,self._ids_only_arms_joints_order[0:4]]
        joints_arm_position_error = torch.square(joints_arm_position - self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order[0:4]])
        joints_arm_position_reward = torch.sum(joints_arm_position_error,dim=1)
    

        rewards = {
            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness_l2": action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt,

            "joints_arm_l2": joints_arm_position_reward * self.cfg.joints_arm_position_reward_scale * self.step_dt,
            "joints_acc_l2": joints_accel * self.cfg.joints_accel_reward_scale * self.step_dt,
            "joints_torques_l2": joints_torques * self.cfg.joints_torque_reward_scale * self.step_dt,
            "joints_energy_l1": joints_energy * self.cfg.joints_energy_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_check_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        died_check_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_ids], dim=-1), dim=1)[0] > 1.0, dim=1) 
        died_arms_collision = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._ids_only_arms_joints_order], dim=-1), dim=1)[0] > 1.0, dim=1)
        died = torch.logical_or(died_check_base, died_check_hips)
        died = torch.logical_or(died, died_arms_collision)
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs: 
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0

        # Sample new commands
        self._ee_commands[env_ids] = torch.zeros_like(self._ee_commands[env_ids]).uniform_(-1.0, 1.0)

        
        # Reset contact periodic
        self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()# + self.step_dt * self._step_freq * torch.rand(env_ids.shape[0], 1, device=self.device)*10.
        self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0

        # Reset noise
        if self.cfg.observation_noise_model:
            self._observation_noise_model_locomotion.reset(env_ids)
        
        if(self.cfg.use_rma):
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma.reset(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = math_utils.random_yaw_orientation(env_ids.shape[0], device=self.device)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())
        
        self.extras["log"].update(extras)



    def _get_new_random_commands(self):
        resample_time = self.episode_length_buf == self.max_episode_length - 300
        commands_resample = torch.zeros_like(self._ee_commands).uniform_(-1.0, 1.0)
        self._ee_commands[:, :3] = self._ee_commands[:, :3] * ~resample_time.unsqueeze(1).expand(-1, 3) + commands_resample * resample_time.unsqueeze(1).expand(-1, 3)

        resample_time_2 = self.episode_length_buf == self.max_episode_length - 600
        commands_resample = torch.zeros_like(self._ee_commands).uniform_(-1.0, 1.0)
        self._ee_commands[:, :3] = self._ee_commands[:, :3] * ~resample_time_2.unsqueeze(1).expand(-1, 3) + commands_resample * resample_time_2.unsqueeze(1).expand(-1, 3)


    def _get_locomotion_policy_action(self,):
        print("TODO")
        return None


    def _get_rma(self, clock_data):
        # Learning privileged information via supervised learning
        obs_rma = torch.cat(
            [
                tensor
                for tensor in (
                    self._ee_commands,
                    self._robot.data.joint_pos[:,self._ids_only_arms_joints_order] - self._robot.data.default_joint_pos[:,self._ids_only_arms_joints_order],
                    self._robot.data.joint_vel[:,self._ids_only_arms_joints_order],
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        #the bottom element is the newest observation!!
        self._observation_history_rma = torch.cat((self._observation_history_rma[:,1:,:], obs_rma.unsqueeze(1)), dim=1)
        obs = torch.flatten(self._observation_history_rma, start_dim=1)

        # Add noise to the observation - this is usually done in direct_rl.py in IsaacLab, but 
        # the obs of cuncurrent SE does not pass from there - its prediciton yes instead!
        if self.cfg.observation_noise_model:          
            obs = self._observation_noise_model_rma(obs.clone())  
        
        outputs_rma = self._get_privileged_observation()

        self._rma_network.dataset.add_sample(obs, outputs_rma)

        # Prediction
        num_episode_from_start = self.common_step_counter / 24. #self.max_episode_length #HACK this should be taken from rsl rl
        num_final_episode_from_start = 8000.
        if num_episode_from_start > self.cfg.rma_ep_saving_interval:
            prediction_rma = self._rma_network(obs)
            obs_rma = prediction_rma
        else:
            obs_rma = outputs_rma

        # Train at some interval
        if num_episode_from_start % self.cfg.rma_ep_saving_interval == 0 and num_episode_from_start > self.cfg.rma_ep_saving_interval - 1:  # Adjust the interval as needed
            self._rma_network.train_network(batch_size=self.cfg.rma_batch_size, 
                                            epochs=self.cfg.rma_train_epochs, 
                                            learning_rate=self.cfg.rma_lr, 
                                            device=self.device)
        if num_episode_from_start == num_final_episode_from_start - 10:
            # Save the network
            self._rma_network.save_network("arm_rma.pth", self.device)
        
        return obs_rma


    def _get_privileged_observation(self):
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*"])
        asset: Articulation = self.scene[asset_cfg.name]
        arm_joints_static_friction = asset.actuators["arm_joint.*"].friction_static

        arm_joints_dynamic_friction = asset.actuators["arm_joint.*"].friction_dynamic

        arm_joints_armature = asset.actuators["arm_joint.*"].armature

        arm_joints_stiffness = asset.actuators["arm_joint.*"].stiffness

        arm_joints_damping = asset.actuators["arm_joint.*"].damping

        default_stiffness = asset.data.default_joint_stiffness[0][0]
        default_damping = asset.data.default_joint_damping[0][0]

        obs_privileged = torch.cat(( 
                            arm_joints_stiffness/default_stiffness, #P gain
                            arm_joints_damping/default_damping, #D gain
                            ) 
                        , dim=-1)
        return obs_privileged