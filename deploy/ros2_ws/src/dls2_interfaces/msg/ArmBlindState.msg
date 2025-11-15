# Header
string frame_id
uint32 sequence_id
float64 timestamp

string robot_name

float64[6] arm_joints_position
float64[6] arm_joints_velocity
float64[6] arm_joints_effort

float64 arm_gripper_position
float64 arm_gripper_velocity
float64 arm_gripper_effort