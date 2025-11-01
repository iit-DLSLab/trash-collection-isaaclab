from enum import Enum

import numpy as np

class ArmStateType(Enum):
    REST = 0
    PREREACH = 1
    REACH = 2
    GRASP = 3
    HOME = 4

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

