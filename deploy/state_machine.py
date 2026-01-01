from enum import Enum

import numpy as np
import copy
import time

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
    def __init__(self):
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


        self.reach_basket_position_1 = np.array([1.61, 1.84, -1.18, 0.01, 0.02, -0.02])
        self.reach_basket_position_2 = np.array([2.17, 1.02, -0.84, -0.71, 1.44, -1.13]) 

        self.open_basket_position_1 = np.array([ 2.74,  0.88, -0.85,  0.2 ,  1.23, -1.85])
        self.open_basket_position_2 = np.array([ 2.74,  0.98, -1.18,  0.06,  1.22, -1.48])

        self.pre_reach_position = np.array([0, 1.5, -1.5, 0.54, 0, 0])

        # Offset for the home position wrt mujoco model
        #self.offset_home_position = np.array([1.27, 0.0, 0.0, 0., 0, 0]) # standard home pos with first joint rotated for the experiment!!
        self.offset_home_position = np.array([0.0, 0.0, 0.0, 0., 0, 0]) # standard home pos with first joint rotated

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
            self.change_state(gripper_state=GripperStateType.CLOSE) # CLOSE

            time_motion = 5.
            initial_joints_position = copy.deepcopy(initial_joints_position)
            reference_joints_position = self.reach_basket_position_1 - self.offset_home_position
            self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
            initial_joints_position = copy.deepcopy(self.desired_position)


        time_motion = 5.
        initial_joints_position = copy.deepcopy(initial_joints_position)
        reference_joints_position = self.home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

        self.change_state(state=ArmStateType.HOME) 


    def armPreReachObject(self, initial_joints_position):
        time_motion = 5.
        initial_joints_position = copy.deepcopy(initial_joints_position)
        reference_joints_position = self.pre_reach_position - self.offset_home_position
        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
        
        self.change_state(state=ArmStateType.PREREACH)

