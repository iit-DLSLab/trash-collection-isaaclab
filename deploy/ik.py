import mujoco
import mujoco.viewer
import numpy as np
import time
from enum import Enum


import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
    
# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-5

# Number of iterations for the IK solver.
num_iterations: int = 150

class IKState(Enum):
    DIFF = 0
    DAMPING = 1
    GRADIENT = 2

class IK:
    def __init__(self) -> None:
        # IK parameters
        self.ik_state = IKState.DAMPING

        # Load model
        self.model = mujoco.MjModel.from_xml_path(dir_path+"/mujoco/models/z1/scene_floating.xml")
        self.data = mujoco.MjData(self.model)

        # Get joint IDs
        joint_names = ["basepitch", "basez", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])

        # Set initial configuration
        key_id = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.final_quat = np.zeros(4)
        self.joint_center = 0.5 * (self.model.jnt_range[:, 0] + self.model.jnt_range[:, 1])
        self.joint_range = 0.5 * (self.model.jnt_range[:, 1] - self.model.jnt_range[:, 0])

        # Get end-effector site ID
        self.site_id = self.model.site("attachment_site").id

    def compute(self, target_pos: np.ndarray, target_quat: np.ndarray, initial_joints_position: np.ndarray, initial_base_pose: np.ndarray, optimize_height = False, optimize_pitch = False) -> np.ndarray:
        
        # Pre-allocate arrays
        ik_succeded = True
        jac = np.zeros((6, self.model.nv))
        diag = damping * np.eye(6)
        error = np.zeros(6)
        error_pos = np.zeros(3)
        error_ori = np.zeros(3)
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)


        # Set initial joint configuration
        self.data.qpos[0:8] = np.concatenate((initial_base_pose, initial_joints_position))
        # I define a new variable to neglect the gripper
        q_eff = np.concatenate((initial_base_pose, initial_joints_position))
        # q = initial_q.copy()
        mujoco.mj_fwdPosition(self.model, self.data)

        for _ in range(num_iterations):
            # Position error
            error_pos = target_pos - self.data.site(self.site_id).xpos

            # Orientation error
            mujoco.mju_mat2Quat(site_quat, self.data.site(self.site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get Jacobian
            mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)


            if(optimize_pitch == False):
                jac[:, 0] = 0.0
            if(optimize_height == False):
                jac[:, 1] = 0.0
            

            # Compute error
            error[:3] = error_pos
            error[3:] = error_ori

            match self.ik_state:
                case IKState.DIFF:
                    # Differential IK
                    dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
                    q = self.data.qpos.copy()
                    q += dq * integration_dt
                    q_eff = q[0:8]

                case IKState.GRADIENT:
                    # Gradient descent
                    q = self.data.qpos.copy()
                    q = q + 0.1 * jac.T @ error
                    q_eff = q[0:8]
            
                case IKState.DAMPING:
                    jac_pseudo = jac.T @ np.linalg.solve(jac @ jac.T + diag, np.eye(6))
                    # Damping to avoid hitting limits
                    dq = jac_pseudo @ error
                    # Damping to avoid joint limits
                    W = np.eye(8)*0.0
                    W[0,0] = 10.
                    W[1,1] = 10.
                    dq -= (np.eye(8) - jac_pseudo @ jac) @ W @ (q_eff - self.joint_center)
                    q_eff += dq * integration_dt    
               
            
            # Update configuration
            self.data.qpos[0:8] = q_eff
            mujoco.mj_fwdPosition(self.model, self.data)
            mujoco.mju_mat2Quat(self.final_quat, self.data.site(self.site_id).xmat)

        if np.linalg.norm(error) > 0.1:
            ik_succeded = False
 
        print("Final IK result:", q_eff)
        print("Success?", ik_succeded)

        # Enforce joint limits
        self.data.qpos = np.clip(self.data.qpos, self.model.jnt_range[self.dof_ids, 0], self.model.jnt_range[self.dof_ids, 1]) 

        final_base_pose = self.data.qpos[0:2] #base pitch, base z
        final_arm_joints = self.data.qpos[2:8]

        return final_base_pose, final_arm_joints, ik_succeded


from scipy.spatial.transform import Rotation as R
if __name__ == "__main__":
    ik_solver = IK()
    
    while True:
        # Define target position and orientation
        x_pos = np.random.uniform(0.2, 0.4)
        y_pos = np.random.uniform(-0.2, 0.2)
        z_pos = np.random.uniform(0.3, 0.6)
        target_pos = np.array([x_pos, y_pos, z_pos])

        roll_grasp = np.random.uniform(-1.8, 1.8)
        pitch_grasp = np.random.uniform(-1.8, 1.8)
        yaw_grasp = np.random.uniform(-1.8, 1.8)
        r = R.from_euler('xyz', [roll_grasp, pitch_grasp, yaw_grasp], degrees=False)
        target_quat = r.as_quat()
        mocap_id = ik_solver.model.body("target").mocapid[0]
        ik_solver.data.mocap_pos[mocap_id] = target_pos
        ik_solver.data.mocap_quat[mocap_id] = target_quat

        # Initial joint configuration
        initial_joints = np.array([0.0, -0.5, 0.5, 0.0, 1.0, 0.0])
        initial_base_pose = np.array([0.0, 0.0])  # pitch, z

        # Compute IK
        final_base_pose, \
        final_arm_joints, \
        success = ik_solver.compute(target_pos, target_quat, initial_joints, initial_base_pose, optimize_height=False, optimize_pitch=True)
        print("IK Success:", success)
        print("Joint Solution:", final_arm_joints)

        # Visualize result
        viewer = mujoco.viewer.launch_passive(ik_solver.model, ik_solver.data)
        while viewer.is_running():
            time.sleep(2)
            breakpoint()
            viewer.close()
            break