# Description: Wrapper of the locomotion policy

# Authors:
# Giulio Turrisi

import time
import copy
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from tqdm import tqdm
import mujoco
import onnxruntime as ort
import torch

import config


import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")

class ManipulationPolicyWrapper:
    def __init__(self, mjModel):

        self.policy = ort.InferenceSession(config.arm_policy_folder_path + "/exported/policy.onnx")
        self.Kp_arm = config.Kp_arm
        self.Kd_arm = config.Kd_arm

        self.RL_FREQ = 1./(config.training_arm_env["sim"]["dt"]*config.training_arm_env["decimation"])  # Hz, frequency of the RL controller


        # RL controller initialization -------------------------------------------------------------
        self.action_scale = config.training_arm_env["action_scale"]
        self.past_rl_actions = np.zeros(8) # only arm
        
        keyframe_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        standUp_qpos = mjModel.key_qpos[keyframe_id]
        self.default_joint_pos_leg = standUp_qpos[7:19]
        self.default_joint_pos_arm = standUp_qpos[19:25]  # default arm joint positions


        # Observation space initialization -------------------------------------------------------
        self.desired_clip_actions = config.training_arm_env["desired_clip_actions"]

        self.use_filter_actions = config.training_arm_env["use_filter_actions"]


        self.use_observation_history = config.training_arm_env["use_observation_history"]
        if(self.use_observation_history):
            single_observation_space = config.training_arm_env["single_observation_space"]
            self.history_length = config.training_arm_env["history_length"]
            self._observation_history = np.zeros((self.history_length, single_observation_space), dtype=np.float32)


        self.use_vision = config.use_vision

        # Desired joint vector
        self.desired_joint_pos = np.zeros(6) # only legs


    def _get_projected_gravity(self, quat_wxyz):        
        # Get the projected gravity in the base frame
        GRAVITY_VEC_W = torch.tensor((0, 0, -9.81), dtype=torch.double)
        GRAVITY_VEC_W = GRAVITY_VEC_W / GRAVITY_VEC_W.norm(p=2, dim=-1).clamp(min=1e-9, max=None).unsqueeze(-1)
        q = torch.tensor(quat_wxyz).view(1, 4)
        v = GRAVITY_VEC_W.clone().detach().view(1, 3)
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        # for two-dimensional tensors, bmm is faster than einsum
        if q_vec.dim() == 2:
            c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
        else:
            c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
        projected_gravity =  a - b + c
        return projected_gravity.numpy().flatten()


    def compute_control(self, 
            base_pos, 
            base_ori_euler_xyz, 
            base_quat_wxyz,
            base_lin_vel, 
            base_ang_vel, 
            heading_orientation_SO3,
            joints_pos_leg, 
            joints_vel_leg,
            ref_base_lin_vel, 
            ref_base_ang_vel,
            joints_pos_arm,
            joints_vel_arm,
            heightmap_data=None):


        # Update Observation ----------------------        
        base_projected_gravity = self._get_projected_gravity(base_quat_wxyz)
        base_vel = base_lin_vel
        base_ang_vel = base_ang_vel

            
        # Fill the observation vector
        joints_pos_delta = joints_pos_leg - self.default_joint_pos_leg
        joints_pos_delta_FL = joints_pos_delta[0:3]
        joints_pos_delta_FR = joints_pos_delta[3:6]
        joints_pos_delta_RL = joints_pos_delta[6:9]
        joints_pos_delta_RR = joints_pos_delta[9:12]

        joints_vel_FL = joints_vel_leg[0:3]
        joints_vel_FR = joints_vel_leg[3:6]
        joints_vel_RL = joints_vel_leg[6:9]
        joints_vel_RR = joints_vel_leg[9:12]


        ee_pose_commands = np.zeros(7)
        ee_pose_commands[0] = 0.0
        ee_pose_commands[1] = 0.2
        ee_pose_commands[2] = 0.3
        
        obs = np.concatenate([
            base_vel, # this could be imu linear acc if use_imu or linear vel from state est
            base_ang_vel,
            base_projected_gravity,
            ee_pose_commands,
            [joints_pos_delta_FL[0]], [joints_pos_delta_FR[0]], [joints_pos_delta_RL[0]], [joints_pos_delta_RR[0]],
            [joints_pos_delta_FL[1]], [joints_pos_delta_FR[1]], [joints_pos_delta_RL[1]], [joints_pos_delta_RR[1]],
            [joints_pos_delta_FL[2]], [joints_pos_delta_FR[2]], [joints_pos_delta_RL[2]], [joints_pos_delta_RR[2]],
            joints_pos_arm - self.default_joint_pos_arm,
            [joints_vel_FL[0]], [joints_vel_FR[0]], [joints_vel_RL[0]], [joints_vel_RR[0]],
            [joints_vel_FL[1]], [joints_vel_FR[1]], [joints_vel_RL[1]], [joints_vel_RR[1]],
            [joints_vel_FL[2]], [joints_vel_FR[2]], [joints_vel_RL[2]], [joints_vel_RR[2]],
            joints_vel_arm,
            self.past_rl_actions.copy(),
        ])

            
        if(self.use_observation_history):
            #the bottom element is the newest observation!!
            past = self._observation_history[1:,:]
            self._observation_history = np.vstack((past, copy.deepcopy(obs)))
            obs = self._observation_history.flatten()


        
        if(self.use_vision):
            # Flatten heightmap with bottom-right at [0], then points going upward
            heightmap_2d = heightmap_data[..., 2][:, :, 0]  # Remove the last dimension
            
            # Flip vertically (so bottom row becomes first) and horizontally (so rightmost becomes first)
            heightmap_flipped = np.flip(heightmap_2d, axis=(0, 1))
            
            # Flatten column-wise so bottom-right is [0], then element above it is [1], etc.
            heightmap_data_isaac_convention = heightmap_flipped.flatten(order='F')

            height_data = (base_pos[2] - heightmap_data_isaac_convention - 0.5)
            height_data = height_data.clip(-1.0, 1.0)
            obs = np.concatenate((obs, height_data), axis=0)
            
        
        # RL Prediction
        obs = obs.reshape(1, -1)
        obs = obs.astype(np.float32)
        rl_action_temp = self.policy.run(None, {'obs': obs})[0][0]
        rl_action_temp = np.clip(rl_action_temp, -self.desired_clip_actions, self.desired_clip_actions)
        

        # Action Filtering
        if(self.use_filter_actions):
            alpha = 0.8
            past_rl_actions_temp = self.past_rl_actions.copy()
            self.past_rl_actions = rl_action_temp.copy()
            rl_action_temp = alpha * rl_action_temp + (1-alpha) * past_rl_actions_temp
        else:
            self.past_rl_actions = rl_action_temp.copy()

        rl_joints_action = rl_action_temp[0:6]
        rl_pose_commands = rl_action_temp[6:8]


        # Impedence Loop
        #TODO fix order of joints in locomotion env
        self.desired_joint_pos = self.default_joint_pos_arm + rl_joints_action*self.action_scale

        
        return self.desired_joint_pos, rl_pose_commands

