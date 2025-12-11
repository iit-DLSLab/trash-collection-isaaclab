import readline
import readchar
import time

import numpy as np
import copy
import mujoco

from state_machine import ArmStateType, GripperStateType

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        self.isDown = True
        self.isRLActivated = False

        # Autocomplete setup
        self.commands = [
            "help", "ictp", "goUp", "goDown", "activate", "ictp", "setKp", "setKd",
            "setBasePose", "armHome", "armPreReachObject", "armReachObjectRL", "armReachObjectIK", "armReachBasket",
            "armOpenBasket", "armCloseGripper", "armOpenGripper"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "goUp"):
                    print("Going Up")
                    if(not self.isDown):
                        print("The robot is already up")
                        continue

                                        
                    start_time = time.time()
                    time_motion = 5.


                    initial_joint_positions = copy.deepcopy(self.controller_node.legs_joints_position)

                    keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
                    standUp_qpos = self.controller_node.mjModel.key_qpos[keyframe_id]
                    reference_joint_positions = standUp_qpos[7:19]

                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        alpha = time_diff / time_motion
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(initial_joint_positions, reference_joint_positions)
                        ]

                        self.controller_node.desired_joint_pos_leg = np.array(interpolated_positions)

                        time.sleep(0.01)

                    self.isDown = False


                elif(input_string == "goDown"):
                    print("Going Down")
                    if(self.isDown):
                        print("The robot is already down")
                        continue

                    self.isDown = True
                    self.isRLActivated = False

                    start_time = time.time()
                    time_motion = 5.

                    temp = copy.deepcopy(self.controller_node.legs_joints_position)
                    initial_joint_positions = temp
                    
                    keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
                    goDown_qpos = self.controller_node.mjModel.key_qpos[keyframe_id]
                    reference_joint_positions = goDown_qpos[7:19]

                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        alpha = time_diff / time_motion
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(initial_joint_positions, reference_joint_positions)
                        ]
            
                        self.controller_node.desired_joint_pos_leg = np.array(interpolated_positions)

                        time.sleep(0.01)

                    
                elif(input_string == "activate"):
                    self.isRLActivated = not self.isRLActivated


                elif(input_string == "help"):
                    self.print_all_commands()


                elif(input_string == "setKp"):
                    print("Kp stand_up_and_down: ", self.controller_node.locomotion_policy.Kp_stand_up_and_down)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.Kp_stand_up_and_down= float(temp)
                    
                    print("Kp walking: ", self.controller_node.locomotion_policy.Kp_walking)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kp_walking = float(temp)
                

                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller_node.locomotion_policy.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.Kd_stand_up_and_down = float(temp)

                    print("Kd walking: ", self.controller_node.locomotion_policy.Kd_walking)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kd_walking = float(temp)
                
                elif(input_string == "ictp"):
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
                    print("1: Pitch Up")
                    print("2: Reset Pitch")
                    print("3: Pitch Down")
                    print("Press any other key to exit")
                    while True:
                        command = readchar.readkey()
                        if(command == "w"):
                            self.controller_node.ref_base_lin_vel_H[0] += 0.1
                            print("w")
                        elif(command == "s"):
                            self.controller_node.ref_base_lin_vel_H[0] -= 0.1
                            print("s")
                        elif(command == "a"):
                            self.controller_node.ref_base_lin_vel_H[1] += 0.1
                            print("a")
                        elif(command == "d"):
                            self.controller_node.ref_base_lin_vel_H[1] -= 0.1
                            print("d")
                        elif(command == "q"):
                            self.controller_node.ref_base_ang_yaw_dot += 0.1
                            print("q")
                        elif(command == "e"):
                            self.controller_node.ref_base_ang_yaw_dot -= 0.1
                            print("e")
                        elif(command == "0"):
                            self.controller_node.ref_base_lin_vel_H[0] = 0
                            self.controller_node.ref_base_lin_vel_H[1] = 0
                            self.controller_node.ref_base_ang_yaw_dot = 0 
                            print("0")
                        elif(command == "1"):
                            self.pitch_delta -= 0.1
                            print("1")
                        elif(command == "2"):
                            self.pitch_delta = 0
                            print("2")
                        elif(command == "3"):
                            self.pitch_delta += 0.1
                            print("3")
                        else:
                            self.controller_node.ref_base_lin_vel_H[0] = 0
                            self.controller_node.ref_base_lin_vel_H[1] = 0
                            self.controller_node.ref_base_ang_yaw_dot = 0 
                            break

                elif input_string == "setBasePose":
                    print("Current Base Position: ", self.controller_node.desired_pose_command)
                    temp = input("Enter Pitch (rad): ")
                    if(temp != ""):
                        self.controller_node.desired_pose_command_overwrite[0] = float(temp)
                    temp = input("Enter Height (m): ")
                    if(temp != ""):
                        self.controller_node.desired_pose_command_overwrite[1] = float(temp)  

                elif input_string =="armHome":
                    start_time = time.time()
                    time_motion = 5.
                    initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                    reference_joints_position = self.controller_node.state_machine.home_position
                    
                    self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)

                    print("Arm in home")
                    self.controller_node.state_machine.change_state(state=ArmStateType.HOME) # REST
                
                elif input_string == "armPreReachObject":
                   
                    start_time = time.time()
                    time_motion = 5.
                    initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                    reference_joints_position = np.array([0, 1.5, -1.5, 0.54, 0, 0]) - self.controller_node.state_machine.offset_home_position

                    self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
                    
                    print("Reached pre-reach")
                    self.controller_node.state_machine.change_state(state=ArmStateType.PREREACH) # Ready for policy handover

                elif input_string == "armReachObjectRL":

                    if(self.controller_node.state_machine.state_type != ArmStateType.PREREACH and self.controller_node.state_machine.state_type != ArmStateType.REACH):
                        print("Error: first move to pre-reach position")
                        continue

                    if(self.controller_node.state_machine.state_type == ArmStateType.PREREACH):
                        self.controller_node.state_machine.change_state(state=ArmStateType.REACH) # Ready for policy handover
                    else:
                        self.controller_node.state_machine.change_state(state=ArmStateType.PREREACH) # Go back in pre-reach
                        start_time = time.time()
                        time_motion = 5.
                        initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                        reference_joints_position = np.array([0, 1.5, -1.5, 0.54, 0, 0]) - self.controller_node.state_machine.offset_home_position

                        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
                        print("Reached pre-reach")

                elif input_string == "armReachObjectIK":

                    if(self.controller_node.state_machine.state_type != ArmStateType.PREREACH and self.controller_node.state_machine.state_type != ArmStateType.REACH):
                        print("Error: first move to pre-reach position")
                        continue
                    
                    target_pos = [0.5, 0.0, 0.1]
                    target_quat = ([ -0.7071, 0.0, -0.7071, 0])
                    print("target pos is ", target_pos)

                    initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                    initial_base_pose = copy.deepcopy(self.controller_node.desired_pose_command_overwrite)
                    
                    reference_base_pose, \
                        reference_joints_position, \
                        ik_succeded = self.controller_node.ik_solver.compute(target_pos, target_quat, initial_joints_position, 
                                                                initial_base_pose, optimize_height=True, optimize_pitch=True)
                    
                    if ik_succeded:
                        
                        time_motion = 2.
                        self.run_base_smoother(initial_base_pose, reference_base_pose, 2.)
                        #self.controller_node.desired_pose_command_overwrite = copy.deepcopy(reference_base_pose)
                        #time.sleep(2)

                        initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                        initial_base_pose = copy.deepcopy(self.controller_node.desired_pose_command_overwrite)
                        
                        _, \
                        reference_joints_position, \
                        ik_succeded = self.controller_node.ik_solver.compute(target_pos, target_quat, initial_joints_position, initial_base_pose)

                        time_motion = 5.
                        self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
                        self.controller_node.state_machine.change_state(state=ArmStateType.GRASP)
                        self.controller_node.state_machine.change_state(gripper_state=GripperStateType.CLOSE) # CLOSE
                        
                    else:
                        print("IK failed, position not reachable!")


                elif input_string == "armReachBasket":

                    start_time = time.time()
                    initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                    reference_joints_position = np.array([1.61, 1.84, -1.18, 0.01, 0.02, -0.02]) - self.controller_node.state_machine.offset_home_position
                    time_motion = 5.

                    self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
                    print("First keypoint reached with joint position: ", self.controller_node.arm_joints_position)

                    start_time = time.time()
                    initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                    reference_joints_position = np.array([2.17, 1.02, -0.84, -0.71, 1.44, -1.13]) - self.controller_node.state_machine.offset_home_position

                    self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)


                elif input_string == "armOpenBasket":

                    self.controller_node.state_machine.change_state(gripper_state=GripperStateType.OPEN) # OPEN

                    start_time = time.time()
                    time_motion = 5.
                    initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                    reference_joints_position = np.array([ 2.74,  0.88, -0.85,  0.2 ,  1.23, -1.85]) - self.controller_node.state_machine.offset_home_position

                    self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)
                    print("First keypoint reached with joint position: ", self.controller_node.arm_joints_position)

                    start_time = time.time()
                    initial_joints_position = copy.deepcopy(self.controller_node.arm_joints_position)
                    reference_joints_position = np.array([ 2.74,  0.98, -1.18,  0.06,  1.22, -1.48]) - self.controller_node.state_machine.offset_home_position
                    
                    self.run_arm_smoother(initial_joints_position, reference_joints_position, time_motion)



                elif input_string == "armCloseGripper":
                    print("Closing gripper")
                    self.controller_node.state_machine.change_state(gripper_state=GripperStateType.CLOSE) # CLOSE


                elif input_string == "armOpenGripper":
                    print("Opening gripper")
                    self.controller_node.state_machine.change_state(gripper_state=GripperStateType.OPEN) # OPEN

            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("ictp: Interactive Keyboard Control")
        print("goUp: Move the robot to standing position")
        print("goDown: Move the robot to crouch position")
        print("activate: Activate/Deactivate RL policy for locomotion\n")
        print("setKp: Set the Kp values for the legs")
        print("setKd: Set the Kd values for the legs")
        print("setBasePose: Set desired base pitch and height")
        print("armHome: Move arm to home position")
        print("armPreReachObject: Move arm to pre-reach object position")
        print("armReachObjectRL: Move arm to reach object position")
        print("armReachObjectIK: Move arm to reach object position using IK")
        print("armReachBasket: Move arm to reach basket position")
        print("armOpenBasket: Open the basket")
        print("armCloseGripper: Close the gripper")
        print("armOpenGripper: Open the gripper\n")


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
            self.controller_node.state_machine.desired_position = interpolated_positions
            time.sleep(0.01)
        print("end of control loop")

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