import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")


locomotion_policy_folder_path = dir_path + "/../tested_policies/locomotion/aliengo_with_z1_last"
arm_policy_folder_path = dir_path + "/../tested_policies/arm/2025-10-30_21-44-02"
# ----------------------------------------------------------------------------------------------------------------

Kp_walking = 25.
Kd_walking = 2.

Kp_stand_up_and_down = 25.
Kd_stand_up_and_down = 2.

Kp_arm = 50.
Kd_arm = 5.

# Load specific training parameters
import yaml 
with open(locomotion_policy_folder_path + "/params/env.yaml", "r") as file:
    training_locomotion_env = yaml.unsafe_load(file)

with open(arm_policy_folder_path + "/params/env.yaml", "r") as file:
    training_arm_env = yaml.unsafe_load(file)

use_vision = False  # If True, use the vision observations in the RL policy
