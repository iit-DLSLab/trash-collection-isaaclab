import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")


locomotion_policy_folder_path = dir_path + "/../tested_policies/locomotion/2026-02-16_12-20-57"
arm_policy_folder_path = dir_path + "/../tested_policies/arm/aliengo_with_z1"
# ----------------------------------------------------------------------------------------------------------------

Kp_walking = 21.5
Kd_walking = 3.5

Kp_stand_up_and_down = 25.
Kd_stand_up_and_down = 2.

Kp_arm = 5.
Kd_arm = 1.

Kp_gripper = 5.
Kd_gripper = 1.

# Load specific training parameters
import yaml 
with open(locomotion_policy_folder_path + "/params/env.yaml", "r") as file:
    training_locomotion_env = yaml.unsafe_load(file)

with open(arm_policy_folder_path + "/params/env.yaml", "r") as file:
    training_arm_env = yaml.unsafe_load(file)

use_vision = False  # If True, use the vision observations in the RL policy
