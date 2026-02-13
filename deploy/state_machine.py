from enum import Enum

import numpy as np
import copy
import time
from scipy.spatial.transform import Rotation

import mujoco

class ArmStateType(Enum):
    REST = 0
    PREREACH = 1
    REACH = 2
    GRASP = 3
    HOME = 4
    BASKET = 5

class GripperStateType(Enum):
    OPEN = 0
    CLOSE = 1

class StateMachine:
    def __init__(self, controller_node):
        self.controller_node = controller_node

        # Number of joints
        self.state_type = ArmStateType.HOME
        self.gripper_state_type = GripperStateType.OPEN

        self.gripper_close_delta = 0.0

        # Pre-defined positions in joint space 
        self.home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # the encoder are relative, so when the robot start is always in zero
        self.desired_position = self.home_position.copy()  # joint pos for unfolded position
        self.desired_velocity = np.zeros(6)
        self.gripper_open_pos = -np.pi/3
        self.gripper_close_pos = 0.1
        self.desired_gripper_position = self.gripper_close_pos #self.gripper_open_pos

        self.arm_rest = np.array([1.8, 1.24, 0.0, -0.4, -1.37, 0.0]) 

        self.reach_basket_position_1 = np.array([1.61, 1.84, -1.18, 0.01, 0.02, -0.02])
        self.reach_basket_position_2 = np.array([2.17, 1.02, -0.84, -0.71, 1.44, -1.13]) 

        self.open_basket_position_1 = np.array([ 2.74,  0.88, -0.85,  0.2 ,  1.23, -1.85])
        self.open_basket_position_2 = np.array([ 2.74,  0.98, -1.18,  0.06,  1.22, -1.48])

        self.pre_reach_position = np.array([0, 1.8, -1.5, 0.54, 0, 0])

        # Offset for the home position wrt mujoco model
        self.offset_home_position = np.array([1.57, 0.0, 0.0, 0., 0, 0]) # standard home pos with first joint rotated for the experiment!!
        #self.offset_home_position = np.array([0.0, 0.0, 0.0, 0., 0, 0]) # standard home pos with first joint rotated

        # Goal end-effector position
        self.goal_ee = None

    def change_state(self, state: ArmStateType = None, gripper_state: GripperStateType = None):
        if state is not None:
            #self.desired_position = self.grasp_position.copy()
            #if state == ArmStateType.REST:
            #    self.desired_position = self.home_position.copy()
            #if state == ArmStateType.GRASP and self.grasp_position is not None:
            #    self.desired_position = self.grasp_position.copy()
            self.state_type = state
        if gripper_state is not None:
            if gripper_state == GripperStateType.OPEN:
                self.desired_gripper_position = self.gripper_open_pos
            elif gripper_state == GripperStateType.CLOSE:
                self.desired_gripper_position = self.gripper_close_pos + self.gripper_close_delta
            self.gripper_state_type = gripper_state

    
    def run_arm_smoother(self , initial_joints_position, reference_joints_position, time_motion):
        start_time = time.time()
        past_joint_positions = copy.deepcopy(initial_joints_position)
        while(time.time() - start_time < time_motion):
            time_diff = time.time() - start_time
            alpha = time_diff / time_motion
            interpolated_positions = [
                (1 - alpha) * initial + alpha * reference
                for initial, reference in zip(initial_joints_position, reference_joints_position)
            ]
            interpolated_positions = np.array(interpolated_positions)
            #interpolated_velocities = (interpolated_positions - past_joint_positions) / 0.01
            past_joint_positions = copy.deepcopy(interpolated_positions)
            self.desired_position = interpolated_positions
            time.sleep(0.01)

    def run_base_smoother(self , initial_base_pose, reference_base_pose, time_motion):
        start_time = time.time()
        past_base_pose = copy.deepcopy(initial_base_pose)
        while(time.time() - start_time < time_motion):
            time_diff = time.time() - start_time
            alpha = time_diff / time_motion
            interpolated_positions = [
                (1 - alpha) * initial + alpha * reference
                for initial, reference in zip(initial_base_pose, reference_base_pose)
            ]
            interpolated_positions = np.array(interpolated_positions)
            #interpolated_velocities = (interpolated_positions - past_joint_positions) / 0.01
            past_base_pose = copy.deepcopy(interpolated_positions)
            self.controller_node.desired_pose_command_overwrite = copy.deepcopy(interpolated_positions)
            time.sleep(0.01)
        print("end of control loop")

    
    def armReachBasket(self, initial_joints_position):
        initial_joints_position = copy.deepcopy(initial_joints_position)
        reference_joints_position = self.reach_basket_position_1 - self.offset_home_position
        time_motion = 5.
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
 
        initial_joints_position = copy.deepcopy(self.desired_position)
        reference_joints_position = self.reach_basket_position_2 - self.offset_home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

        self.change_state(state=ArmStateType.BASKET)


    def armOpenBasket(self, initial_joints_position):

        if(self.state_type != ArmStateType.BASKET):
            print("Go to BASKET position first!")
            return
        
        self.change_state(gripper_state=GripperStateType.OPEN) # OPEN

        time_motion = 5.
        initial_joints_position = copy.deepcopy(initial_joints_position)
        reference_joints_position = self.open_basket_position_1 - self.offset_home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)


        initial_joints_position = copy.deepcopy(self.desired_position)
        reference_joints_position = self.open_basket_position_2 - self.offset_home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)


    def armHome(self, initial_joints_position):
        if(self.state_type == ArmStateType.BASKET):
            time_motion = 5.
            initial_joints_position = copy.deepcopy(initial_joints_position)
            reference_joints_position = self.open_basket_position_1 - self.offset_home_position
            self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

            initial_joints_position = copy.deepcopy(self.desired_position)
            reference_joints_position = self.reach_basket_position_1 - self.offset_home_position
            self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

            initial_joints_position = copy.deepcopy(self.desired_position)

            self.change_state(gripper_state=GripperStateType.CLOSE) # CLOSE


        time_motion = 5.
        initial_joints_position = copy.deepcopy(initial_joints_position)
        reference_joints_position = self.home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

        self.change_state(state=ArmStateType.HOME) 


    def armRest(self, initial_joints_position):
        if(self.state_type == ArmStateType.BASKET):
            time_motion = 5.
            initial_joints_position = copy.deepcopy(initial_joints_position)
            reference_joints_position = self.open_basket_position_1 - self.offset_home_position
            self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

            initial_joints_position = copy.deepcopy(self.desired_position)
            reference_joints_position = self.reach_basket_position_1 - self.offset_home_position
            self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

            initial_joints_position = copy.deepcopy(self.desired_position)

            self.change_state(gripper_state=GripperStateType.CLOSE) # CLOSE


        time_motion = 5.
        initial_joints_position = copy.deepcopy(initial_joints_position)
        reference_joints_position = self.arm_rest - self.offset_home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

        self.change_state(state=ArmStateType.HOME) 


    def armPreReachObject(self, initial_joints_position):
        time_motion = 5.
        initial_joints_position = copy.deepcopy(initial_joints_position)
        reference_joints_position = self.pre_reach_position - self.offset_home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
        
        self.change_state(state=ArmStateType.PREREACH)

    
    def armReachObjectIK(self, initial_joints_position):
        if(self.state_type != ArmStateType.PREREACH and self.state_type != ArmStateType.REACH):
            print("Error: first move to pre-reach position")
            return

        self.controller_node.state_machine.change_state(gripper_state=GripperStateType.OPEN) # OPEN

        if(self.controller_node.received_detection):
            self.controller_node.received_detection = False
        else:
            print("No detection received, cannot reach object!")
            return

        target_pos, target_quat = self.detection_from_camera_to_base()
        initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position) + self.offset_home_position
        initial_base_pose = copy.deepcopy(self.controller_node.desired_pose_command_overwrite)
        initial_base_pose_placeholder = copy.deepcopy(initial_base_pose)
        
        reference_base_pose, \
            reference_joints_position, \
            ik_succeded = self.controller_node.ik_mink_solver.compute(target_pos, target_quat, initial_joints_position, 
                                                    initial_base_pose, optimize_height=True, optimize_pitch=True)
        
        if ik_succeded:
            # First move the base
            self.run_base_smoother(initial_base_pose, reference_base_pose, time_motion=2.)

            # Then move the arm in two steps, reaching an intermediate point
            target_pos, target_quat = self.detection_from_camera_to_base()
            initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position) + self.offset_home_position
            initial_base_pose = copy.deepcopy(self.controller_node.desired_pose_command_overwrite)
            intermediate_target_pos = copy.deepcopy(target_pos)
            #I need to apply an offset in the direction of grasping
            offset_end_effector_frame = np.array([-0.15, 0.0, 0.0])
            R_TE = np.zeros((3, 3), dtype=float)
            mujoco.mju_quat2Mat(R_TE.reshape(9,), target_quat)
            R_TE = R_TE.reshape(3,3)
            offset_base_frame = R_TE @ offset_end_effector_frame

            intermediate_target_pos += offset_base_frame
            _, \
                reference_joints_position, \
                ik_succeded = self.controller_node.ik_mink_solver.compute(intermediate_target_pos, target_quat, initial_joints_position, initial_base_pose)
            
            if ik_succeded:
                self.run_arm_smoother(initial_joints_position, reference_joints_position - self.offset_home_position, time_motion=3.)

                # Finally reach the target
                initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position) + self.offset_home_position
                initial_base_pose = copy.deepcopy(self.controller_node.desired_pose_command_overwrite)

                #Other offset to grasp better in the direction of grasping
                offset_end_effector_frame = np.array([0.05, 0.0, 0.0])
                R_TE = np.zeros((3, 3), dtype=float)
                mujoco.mju_quat2Mat(R_TE.reshape(9,), target_quat)
                R_TE = R_TE.reshape(3,3)
                offset_base_frame = R_TE @ offset_end_effector_frame
                target_pos += offset_base_frame
                
                _, \
                    reference_joints_position, \
                    ik_succeded = self.controller_node.ik_mink_solver.compute(target_pos, target_quat, initial_joints_position, initial_base_pose)
                
                if ik_succeded:
                    self.run_arm_smoother(initial_joints_position, reference_joints_position - self.offset_home_position, time_motion=2.)

                    # Close the gripper and grasp
                    time.sleep(1.)
                    self.change_state(state=ArmStateType.GRASP)
                    self.change_state(gripper_state=GripperStateType.CLOSE) # CLOSE

                    # Return to previous base position base position
                    self.run_base_smoother(initial_base_pose, initial_base_pose_placeholder, time_motion = 2.)
            
        else:
            print("IK failed, position not reachable!")

    
    def armReachObjectRL(self, initial_joints_position):
        if(self.state_type != ArmStateType.PREREACH and self.state_type != ArmStateType.REACH):
            print("Error: first move to pre-reach position")
            return

        if(self.state_type == ArmStateType.PREREACH):
            self.state_type = ArmStateType.REACH # Ready for policy handover
        else:
            self.state_type = ArmStateType.PREREACH # Go back in pre-reach
            start_time = time.time()
            time_motion = 5.
            initial_joints_position = copy.deepcopy(initial_joints_position)
            reference_joints_position = self.pre_reach_position - self.offset_home_position

            self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
            print("Reached pre-reach")


    def detection_from_camera_to_base(self, ):

        # Trasformation from camera frame to base frame
        body_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        cam_id  = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_CAMERA, "robotcam")

        p_WB = self.controller_node.mjData.xpos[body_id]
        R_WB = self.controller_node.mjData.xmat[body_id].reshape(3, 3)

        p_WC = self.controller_node.mjData.cam_xpos[cam_id]
        R_WC = self.controller_node.mjData.cam_xmat[cam_id].reshape(3, 3)

        R_BC = R_WB.T @ R_WC
        t_BC = R_WB.T @ (p_WC - p_WB)

        # --- position ---
        p_CO = self.controller_node.ik_goal_camera_frame
        p_B = R_BC @ p_CO + t_BC

        # --- orientation ---
        # Convert q_C to rotation matrix
        R_CO = np.zeros((3, 3), dtype=float)
        mujoco.mju_quat2Mat(R_CO.reshape(9,), self.controller_node.ik_goal_orient_camera_frame)
        R_CO = R_CO.reshape(3,3)
        # Compose: body_R = (body<-camera) * (camera_R)
        R_BO = R_BC @ R_CO

        # Convert back to quaternion (w,x,y,z)
        q_BO = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(q_BO, R_BO.reshape(9,))
        R_BO = R_BO.reshape(3,3)

        #transform in RPY and remove roll and yaw
        #rpy = np.zeros(3)
        #rpy[1] = np.arcsin(-R_BO[2, 0]) 
        #q_target = np.zeros(4, dtype=float)
        #mujoco.mju_euler2Quat(q_target, rpy, "XYZ")
        #return p_B, q_target

        #quat_wxyz = q_BO
        #quat_xyzw = np.roll(quat_wxyz, -1)
        #rpy = Rotation.from_quat(quat_xyzw).as_euler('xyz')
        #rpy[2] = 0.0
        #mujoco.mju_euler2Quat(q_target, rpy, "XYZ")
        #return p_B, q_target


        return p_B, q_BO

