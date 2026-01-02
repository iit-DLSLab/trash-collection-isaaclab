# Description: This script is used to run the policy on the real robot

# Authors:
# Giulio Turrisi

import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Joy
from dls2_interface.msg import BaseState, BlindState, TrajectoryGenerator, ArmState, ArmTrajectoryGenerator, ArmControlSignal
from geometry_msgs.msg import PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy

import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/mujoco/")
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

# Simulation related imports
import mujoco
import mujoco.viewer
import mujoco_utils
from heightmap import HeightMap

# Trash Policy imports
from manipulation_policy_wrapper import ManipulationPolicyWrapper
from locomotion_policy_wrapper import LocomotionPolicyWrapper
from state_machine import StateMachine

import config
import threading

# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


class TrashControlNode(Node):
    def __init__(self):
        super().__init__('Trash_Control_Node')

        self.simulation_dt = 0.002

        # Load the model and data.
        self.mjModel = mujoco.MjModel.from_xml_path("mujoco/models/scene_rough.xml")
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]


        # Initialization of variables used in the main control loop --------------------------------
        self.manipulation_policy = ManipulationPolicyWrapper(mjModel=self.mjModel)
        self.locomotion_policy = LocomotionPolicyWrapper(mjModel=self.mjModel)
        self.state_machine = StateMachine()

        if(self.locomotion_policy.use_vision):
            resolution_heightmap = config.resolution_heightmap
            num_rows_heightmap = round(config.size_x_heightmap/resolution_heightmap) + 1
            num_cols_heightmap = round(config.size_y_heightmap/resolution_heightmap) + 1
            self.heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=mjModel, mj_data=mjData) 

        self.arm_joints_position = np.zeros(6)  # 6 arm joints 
        self.arm_joints_velocity = np.zeros(6)  # 6 arm joints
        self.legs_joints_position = np.zeros(12)  # 12 leg joints
        self.legs_joints_velocity = np.zeros(12)  # 12 leg joints 
        self.desired_joint_pos_arm = self.mjData.qpos[19:25] 
        self.desired_joint_pos_leg = self.mjData.qpos[7:19]
        self.desired_pose_command = np.zeros(2)
        self.desired_pose_command_overwrite = np.zeros(2)
        self.Kp_legs = 0
        self.Kd_legs = 0
        self.Kp_arm = 0
        self.Kd_arm = 0
        

        # --------------------------------------------------------------
        self.ref_base_lin_vel_H = np.array([0.0, 0.0, 0.0])  # Desired base linear velocity in the horizontal plane (x, y, z)
        self.ref_base_ang_yaw_dot = 0.0  # Desired base angular velocity around the vertical axis

        # Interactive Command Line
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

        #self.console.isDown = True  # Only in this play_mujoco script
        #self.console.isRLActivated = False  # Only in this play_mujoco script
                 
        # --------------------------------------------------------------
        # Subscribers and Publishers
        self.subscription_base_state = self.create_subscription(BaseState,"/base_state", self.get_base_state_callback, 1)
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state", self.get_blind_state_callback, 1)
        self.subscription_arm_blind_state = self.create_subscription(ArmState,"/arm_state", self.get_arm_blind_state_callback, 1)
        self.subscription_joy = self.create_subscription(Joy,"/joy", self.get_joy_callback, 1)
        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGenerator,"/trajectory_generator", 1)
        self.publisher_arm_trajectory_generator = self.create_publisher(ArmTrajectoryGenerator,"/arm_trajectory_generator", 1)
        self.publisher_arm_control_signal = self.create_publisher(ArmControlSignal,"/arm_control_signal", 1)
        RL_FREQ = 1./(config.training_locomotion_env["sim"]["dt"]*config.training_locomotion_env["decimation"])  # Hz, frequency of the RL controller
        self.timer = self.create_timer(1.0/RL_FREQ, self.compute_rl_control)

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.subscription_orientation = self.create_subscription(PoseArray,"detections3d/grasp_poses", self.get_grasp_pose_callback, qos_profile)

        # Variables for IK from detection
        self.received_detection = False
        self.ik_goal_orient_camera_frame = np.array([1.0, 0.0 ,0.0 ,0.0])
        self.r_optical_to_camera_frame = np.array([
                                                [0.0, 0.0,  1.0],
                                                [-1.0, 0.0, 0.0],
                                                [0.0, -1.0, 0.0]
                                            ])
        self.r_camera_to_base_frame = np.array([
                                                [1.0, 0.0,  0.0],
                                                [0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0]
                                            ])

        self.r_camera_frame_to_base = self.r_camera_to_base_frame @ self.r_optical_to_camera_frame

        self.ik_goal_camera_frame = np.zeros(3)
        self.ik_goal_base_frame = np.zeros(3)
        self.ik_goal_orient_base_frame = np.array([1.0, 0.0 ,0.0 ,0.0])

        # Safety check to not do anything until a first base and blind state are received
        self.first_message_base_arrived = False
        self.first_message_legs_joints_arrived = False
        self.first_message_arm_joints_arrived = True
        self.last_joy_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)


    def get_grasp_pose_callback(self, msg):
        if len(msg.poses) > 0:
            pose = msg.poses[0]
            self.ik_goal_camera_frame[0] = pose.position.x
            self.ik_goal_camera_frame[1] = pose.position.y
            self.ik_goal_camera_frame[2] = pose.position.z

            #conversion from camera optical frame to base
            self.ik_goal_base_frame =  self.r_camera_frame_to_base @ self.ik_goal_camera_frame

            self.ik_goal_orient_camera_frame[0] = pose.orientation.w
            self.ik_goal_orient_camera_frame[1] = pose.orientation.x
            self.ik_goal_orient_camera_frame[2] = pose.orientation.y
            self.ik_goal_orient_camera_frame[3] = pose.orientation.z

            self.ik_goal_orient_base_frame = ( R.from_matrix(self.r_camera_frame_to_base) * R.from_quat(self.ik_goal_orient_camera_frame, scalar_first=True)).as_quat(scalar_first=True)
            self.received_detection = True

    
    def get_joy_callback(self, msg):
        """
        Callback function to handle joystick input. Joystick used is a 
        8Bitdi Ultimate 2C Wireless Controller.
        """
        filter_joystick = 0.7
        self.ref_base_lin_vel_H[0] = self.ref_base_lin_vel_H[0]*filter_joystick + (msg.axes[1]/3.5)*(1-filter_joystick)  # Forward/Backward
        self.ref_base_lin_vel_H[1] = self.ref_base_lin_vel_H[1]*filter_joystick + (msg.axes[0]/3.5)*(1-filter_joystick)  # Left/Right
        self.ref_base_ang_yaw_dot = self.ref_base_ang_yaw_dot*filter_joystick + (msg.axes[3]/2.)*(1-filter_joystick)  # Yaw

        self.last_joy_time = time.time()


        #kill the node if the button is pressed
        if msg.buttons[8] == 1:
            self.get_logger().info("Joystick button pressed, shutting down the node.") 
            # This will kill the robot hal
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            # This will kill the process running this script
            os.system("pkill -f play_ros2.py") 
            exit(0)


    def get_base_state_callback(self, msg):
        self.position = np.array(msg.pose.position) #world frame
        # For the quaternion, the order is [w, x, y, z] on mujoco, and [x, y, z, w] on DLS2
        self.orientation = np.roll(np.array(msg.pose.orientation), 1) #world frame
        self.linear_velocity = np.array(msg.velocity.linear) #world frame
        self.angular_velocity = np.array(msg.velocity.angular) #base frame

        self.first_message_base_arrived = True


    def get_blind_state_callback(self, msg):
        self.legs_joints_position = np.array(msg.joints_position)
        self.legs_joints_velocity = np.array(msg.joints_velocity)

        # Fix convention DLS2
        self.legs_joints_position[0] = -self.legs_joints_position[0]
        self.legs_joints_position[6] = -self.legs_joints_position[6]
        self.legs_joints_velocity[0] = -self.legs_joints_velocity[0]
        self.legs_joints_velocity[6] = -self.legs_joints_velocity[6]

        self.first_message_legs_joints_arrived = True


    def get_arm_blind_state_callback(self, msg):        
        self.arm_joints_position = np.array(msg.joints_position)
        self.arm_joints_velocity = np.array(msg.joints_velocity)

        self.first_message_arm_joints_arrived = True


    def compute_rl_control(self):        
        # Safety check to not do anything until a first base and blind state are received
        if(self.first_message_base_arrived==False or self.first_message_legs_joints_arrived==False or self.first_message_arm_joints_arrived==False):
            return
        
        # Safety check for joystick timeout
        if(self.last_joy_time is not None and time.time() - self.last_joy_time > 1.0):
            self.ref_base_lin_vel_H[0] = 0.0
            self.ref_base_lin_vel_H[1] = 0.0
            self.ref_base_ang_yaw_dot = 0.0
            print("Joystick timeout, stopping the robot")
            self.last_joy_time = None
    

        # Update the mujoco model
        self.mjData.qpos[0:3] = copy.deepcopy(self.position)
        self.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)

        self.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
        self.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
        
        self.mjData.qpos[7:19] = copy.deepcopy(self.legs_joints_position)
        self.mjData.qvel[6:18] = copy.deepcopy(self.legs_joints_velocity)
        self.mjData.qpos[19:25] = copy.deepcopy(self.arm_joints_position)
        self.mjData.qvel[18:24] = copy.deepcopy(self.arm_joints_velocity)
        mujoco.mj_forward(self.mjModel, self.mjData)


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


        if(self.console.isDown):
            # Impedence Loop
            self.Kp_legs = self.locomotion_policy.Kp_stand_up_and_down
            self.Kd_legs = self.locomotion_policy.Kd_stand_up_and_down
            self.Kp_arm = self.manipulation_policy.Kp_arm
            self.Kd_arm = self.manipulation_policy.Kd_arm

        # RL controller --------------------------------------------------------------
        if self.console.isRLActivated:            
            
            """if self.state_machine.state_type == ArmStateType.REACH:
                self.desired_joint_pos_arm, self.desired_pose_command = self.manipulation_policy.compute_control(
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
            else:
                self.desired_joint_pos_arm = self.state_machine.desired_position
                self.desired_pose_command = np.zeros(2)"""


            self.desired_joint_pos_leg = self.locomotion_policy.compute_control(
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
                        ref_pose_command=self.desired_pose_command + self.desired_pose_command_overwrite,
                        heightmap_data=self.heightmap.data if self.locomotion_policy.use_vision else None)

            self.Kp_legs = self.locomotion_policy.Kp_walking
            self.Kd_legs = self.locomotion_policy.Kd_walking
            self.Kp_arm = self.manipulation_policy.Kp_arm
            self.Kd_arm = self.manipulation_policy.Kd_arm

        
        self.desired_joint_pos_arm = self.state_machine.desired_position

        
        # Fix convention DLS2 and send PD target
        self.desired_joint_pos_leg[0] = -self.desired_joint_pos_leg[0]
        self.desired_joint_pos_leg[6] = -self.desired_joint_pos_leg[6]

            
        trajectory_generator_msg = TrajectoryGenerator()
        trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        trajectory_generator_msg.joints_position = self.desired_joint_pos_leg.tolist()
        trajectory_generator_msg.joints_velocity = np.zeros(12).tolist()
        trajectory_generator_msg.kp = (np.ones(12)*self.Kp_legs).tolist()
        trajectory_generator_msg.kd = (np.ones(12)*self.Kd_legs).tolist()
        self.publisher_trajectory_generator.publish(trajectory_generator_msg)

        arm_trajectory_generator_msg = ArmTrajectoryGenerator()
        arm_trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        arm_trajectory_generator_msg.desired_arm_joints_position = self.desired_joint_pos_arm.tolist()
        arm_trajectory_generator_msg.desired_arm_joints_velocity = np.zeros(6).tolist()
        arm_trajectory_generator_msg.arm_kp = (np.ones(6)*self.Kp_arm).tolist()
        arm_trajectory_generator_msg.arm_kd = (np.ones(6)*self.Kd_arm).tolist()
        self.publisher_arm_trajectory_generator.publish(arm_trajectory_generator_msg)

        arm_control_signal_msg = ArmControlSignal()
        arm_control_signal_msg.desired_arm_joints_torque = self.mjData.qfrc_bias[18:24].tolist()  # Send the gravity compensation torques
        arm_control_signal_msg.desired_arm_gripper_torque = 0.0  # Placeholder for gripper torque
        self.publisher_arm_control_signal.publish(arm_control_signal_msg)


#---------------------------
if __name__ == '__main__':
    
    print('Hello from the trash control ros node.')
    
    rclpy.init()
    trash_control_node = TrashControlNode()
    rclpy.spin(trash_control_node)
    trash_control_node.destroy_node()
    rclpy.shutdown()

    print("trash control ros node is stopped")
    exit(0)