# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Giulio Turrisi

import time
import numpy as np
from tqdm import tqdm
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/mujoco/")
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

# Gym and Simulation related imports
import mujoco
import mujoco.viewer
import mujoco_utils
from heightmap import HeightMap


# Trash Policy imports
from manipulation_policy_wrapper import ManipulationPolicyWrapper
from locomotion_policy_wrapper import LocomotionPolicyWrapper

import config
import threading


class PlayMujoco:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)

        self.simulation_dt = 0.002


        # Load the model and data.
        self.mjModel = mujoco.MjModel.from_xml_path("mujoco/models/scene_rough.xml")
        self.mjData = mujoco.MjData(self.mjModel)


        # Initialization of variables used in the main control loop --------------------------------
        self.manipulation_policy = ManipulationPolicyWrapper(mjModel=self.mjModel)
        self.locomotion_policy = LocomotionPolicyWrapper(mjModel=self.mjModel)

        if(self.locomotion_policy.use_vision):
            resolution_heightmap = config.resolution_heightmap
            num_rows_heightmap = round(config.size_x_heightmap/resolution_heightmap) + 1
            num_cols_heightmap = round(config.size_y_heightmap/resolution_heightmap) + 1
            self.heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=mjModel, mj_data=mjData)    
        

        # --------------------------------------------------------------
        self.ref_base_lin_vel_H = np.array([0.0, 0.0, 0.0])  # Desired base linear velocity in the horizontal plane (x, y, z)
        self.ref_base_ang_yaw_dot = 0.0  # Desired base angular velocity around the vertical axis

        # Interactive Command Line
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()


    def run(self):
        # Run the simulation
        step_num = 0
        RENDER_FREQ = 30  # Hz
        last_render_time = time.time()
        with mujoco.viewer.launch_passive(
            model=self.mjModel,
            data=self.mjData,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # Get the current state of the robot -----------------------------------------------------
                qpos, qvel = self.mjData.qpos, self.mjData.qvel
                base_lin_vel = mujoco_utils.base_lin_vel(self.mjData, frame='base')
                base_ang_vel = mujoco_utils.base_ang_vel(self.mjData, frame='base')
                base_ori_euler_xyz = mujoco_utils.base_ori_euler_xyz(self.mjData)
                heading_orientation_SO3 = mujoco_utils.heading_orientation_SO3(self.mjData)
                base_quat_wxyz = qpos[3:7]
                base_pos = mujoco_utils.base_pos(self.mjData)

                joints_pos_leg = qpos[7:19]
                joints_pos_arm = qpos[19:25]
                joints_pos_gripper = qpos[25]

                joints_vel_leg = qvel[6:18]
                joints_vel_arm = qvel[18:24]
                joints_vel_gripper = qvel[24]

            
                ref_base_lin_vel, ref_base_ang_vel = mujoco_utils.target_base_vel(self.mjData, self.ref_base_lin_vel_H, self.ref_base_ang_yaw_dot, frame='world')


                if(self.locomotion_policy.use_vision):
                    self.heightmap.update_height_map(self.mjData.qpos[0:3], yaw=base_ori_euler_xyz[2])
            
                # RL controller --------------------------------------------------------------
                if step_num % round(1 / (self.locomotion_policy.RL_FREQ * self.simulation_dt)) == 0:            
                    
                    desired_joint_pos_arm, pose_commands = self.manipulation_policy.compute_control(
                                base_pos=base_pos, 
                                base_ori_euler_xyz=base_ori_euler_xyz, 
                                base_quat_wxyz=base_quat_wxyz,
                                base_lin_vel=base_lin_vel, 
                                base_ang_vel=base_ang_vel,
                                heading_orientation_SO3=heading_orientation_SO3,
                                joints_pos_leg=joints_pos_leg, 
                                joints_vel_leg=joints_vel_leg,
                                joints_pos_arm=joints_pos_arm,
                                joints_vel_arm=joints_vel_arm,
                                ref_base_lin_vel=ref_base_lin_vel, 
                                ref_base_ang_vel=ref_base_ang_vel,
                                heightmap_data=self.heightmap.data if self.locomotion_policy.use_vision else None)

                    desired_joint_pos_leg = self.locomotion_policy.compute_control(
                                base_pos=base_pos, 
                                base_ori_euler_xyz=base_ori_euler_xyz, 
                                base_quat_wxyz=base_quat_wxyz,
                                base_lin_vel=base_lin_vel, 
                                base_ang_vel=base_ang_vel,
                                heading_orientation_SO3=heading_orientation_SO3,
                                joints_pos_leg=joints_pos_leg, 
                                joints_vel_leg=joints_vel_leg,
                                joints_pos_arm=joints_pos_arm,
                                ref_base_lin_vel=ref_base_lin_vel, 
                                ref_base_ang_vel=ref_base_ang_vel,
                                heightmap_data=self.heightmap.data if self.locomotion_policy.use_vision else None)

                # PD controller --------------------------------------------------------------
                else:
                    desired_joint_pos_leg = self.locomotion_policy.desired_joint_pos
                    desired_joint_pos_arm = self.manipulation_policy.desired_joint_pos


                error_joints_pos_leg = desired_joint_pos_leg - joints_pos_leg
                tau_leg = self.locomotion_policy.Kp_walking*error_joints_pos_leg - self.locomotion_policy.Kd_walking*joints_vel_leg

                error_joints_pos_arm = desired_joint_pos_arm - joints_pos_arm
                tau_arm = self.manipulation_policy.Kp_arm*error_joints_pos_arm - self.manipulation_policy.Kd_arm*joints_vel_arm


                # Set control and mujoco step ----------------------------------------------------------------------
                self.mjData.ctrl[0:12] = tau_leg
                self.mjData.ctrl[12:18] = tau_arm
                mujoco.mj_step(self.mjModel, self.mjData)
                step_num = step_num +1


                # Sleep to match real-time ---------------------------------------------------------
                loop_elapsed_time = time.time() - step_start
                if(loop_elapsed_time < self.simulation_dt):
                    time.sleep(self.simulation_dt - (loop_elapsed_time))


                # Render only at a certain frequency -----------------------------------------------------------------
                if time.time() - last_render_time > 1.0 / RENDER_FREQ or step_num == 1:

                    # Update the camera position
                    viewer.cam.lookat[:] = base_pos
                    
                    # Draw other stuff if needed
                    if(self.locomotion_policy.use_vision):
                        print("Draw to made")
                    
                    viewer.sync()
                    last_render_time = time.time()


if __name__ == "__main__":
    play_mujoco = PlayMujoco()
    play_mujoco.run()