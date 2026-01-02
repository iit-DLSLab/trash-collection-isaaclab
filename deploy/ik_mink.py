from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib.keyboard_teleop import keycodes

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")

import time
from scipy.spatial.transform import Rotation as R

"""@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause"""

class IKMink:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(dir_path+"/mujoco/models/z1/scene_floating.xml")
        self.data = mujoco.MjData(self.model)

        # Joints we wish to control.
        # fmt: off
        self.joint_names = [
            "basepitch", "basez", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]
        # fmt: on
        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        #self.actuator_ids = np.array([self.model.actuator(name).id for name in self.joint_names])

        self.configuration = mink.Configuration(self.model)

        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )

        self.posture_cost = np.zeros((self.model.nv,))
        self.posture_cost[2:] = 1e-3
        self.posture_task = mink.PostureTask(self.model, cost=self.posture_cost)

        self.immobile_base_cost = np.zeros((self.model.nv,))
        self.immobile_base_cost[0:2] = 100
        self.damping_task = mink.DampingTask(self.model, self.immobile_base_cost)

        self.tasks = [
            self.end_effector_task,
            self.posture_task,
        ]

        # Enable collision avoidance between the following geoms.
        # left hand - table, right hand - table
        # left hand - left thigh, right hand - right thigh
        self.collision_pairs = [
            (["link02_geom2"], ["trunk"]),
        ]
        self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=self.collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.005,
            collision_detection_distance=0.15,
        )

        self.limits = [
            mink.ConfigurationLimit(self.model),
            self.collision_avoidance_limit,
        ]

        # IK settings.
        self.solver = "daqp"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 20

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(self.model, self.data, "target", "attachment_site", "site")

    def compute(self, target_pos: np.ndarray, target_quat: np.ndarray, initial_joints_position: np.ndarray, initial_base_pose: np.ndarray, optimize_height = False, optimize_pitch = False) -> np.ndarray:
        
        self.data.qpos[0:8] = np.concatenate((initial_base_pose, initial_joints_position))
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Update task target.
        mocap_id = self.model.body("target").mocapid[0]
        self.data.mocap_pos[mocap_id] = target_pos
        self.data.mocap_quat[mocap_id] = target_quat

        T_wt = mink.SE3.from_mocap_name(self.model, self.data, "target")
        self.end_effector_task.set_target(T_wt)

        # Compute velocity and integrate into the next configuration.
        for i in range(self.max_iters):
            vel = mink.solve_ik(
                self.configuration,
                [*self.tasks, self.damping_task],
                0.005,
                self.solver,
                damping=1e-3,
            )

            #vel = mink.solve_ik(
            #    self.configuration, self.tasks, 0.005, self.solver, damping=1e-3
            #)
            
            self.configuration.integrate_inplace(vel, 0.005)

            # Exit condition.
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = bool(np.linalg.norm(err[:3]) <= self.pos_threshold)
            ori_achieved = bool(np.linalg.norm(err[3:]) <= self.ori_threshold)
            if pos_achieved and ori_achieved:
                ik_succeded = True
            else:
                ik_succeded = False

        final_base_pose = self.configuration.q[0:2] #base pitch, base z
        final_arm_joints = self.configuration.q[2:8]
        
        return final_base_pose, final_arm_joints, ik_succeded


if __name__ == "__main__":
    
    ik_solver = IKMink()
    
    step = 0

    # Initial joint configuration
    initial_joints = np.array([0.0, -0.5, 0.5, 0.0, 1.0, 0.0])
    initial_base_pose = np.array([0.0, 0.0])  # pitch, z

    # Visualize result
    viewer = mujoco.viewer.launch_passive(ik_solver.model, ik_solver.data)
    while viewer.is_running():

        if(step % 100 == 0):
            # Define target position and orientation
            x_pos = np.random.uniform(0.4, 0.4)
            y_pos = np.random.uniform(-0.2, 0.2)
            z_pos = np.random.uniform(0.3, 0.6)
            target_pos = np.array([x_pos, y_pos, z_pos])

            roll_grasp = np.random.uniform(-1.8, 1.8)
            pitch_grasp = np.random.uniform(-1.8, 1.8)
            yaw_grasp = np.random.uniform(-1.8, 1.8)
            r = R.from_euler('xyz', [roll_grasp, pitch_grasp, yaw_grasp], degrees=False)
            target_quat = r.as_quat()
            breakpoint()


        # Compute IK
        final_base_pose, \
        final_arm_joints, \
        success = ik_solver.compute(target_pos, target_quat, initial_joints, initial_base_pose, optimize_height=False, optimize_pitch=True)

        # Update initial conditions for the next iteration
        initial_joints = final_arm_joints
        initial_base_pose = final_base_pose

        ik_solver.data.qpos[0:8] = np.concatenate((initial_base_pose, initial_joints))

        mujoco.mj_kinematics(ik_solver.model, ik_solver.data)

        viewer.sync()
        step += 1
        print("step: ", step)
        time.sleep(0.005)
