diff --git a/source/trash_collection_isaaclab/trash_collection_isaaclab/assets/aliengo_asset.py b/source/trash_collection_isaaclab/trash_collection_isaaclab/assets/aliengo_asset.py
index 94cc0ad..e69de29 100644
--- a/source/trash_collection_isaaclab/trash_collection_isaaclab/assets/aliengo_asset.py
+++ b/source/trash_collection_isaaclab/trash_collection_isaaclab/assets/aliengo_asset.py
@@ -1,149 +0,0 @@
-# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
-# All rights reserved.
-#
-# SPDX-License-Identifier: BSD-3-Clause
-
-import isaaclab.sim as sim_utils
-from trash_collection_isaaclab.actuators import IdentifiedActuatorElectricCfg
-from isaaclab.actuators import DelayedPDActuatorCfg
-from isaaclab.assets.articulation import ArticulationCfg
-
-from trash_collection_isaaclab.assets import ISAAC_ASSET_DIR
-
-# Aliengo robot configuration from mujoco
-stiffness_mujoco = 25.0
-damping_mujoco = 2.0
-
-"""
-friction_static_mujoco = 0.2
-friction_dynamic_mujoco = 0.6
-armature_mujoco = 0.01
-
-ALIENGO_HIP_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
-    joint_names_expr=[".*_hip_joint"],
-    effort_limit=44.4,
-    velocity_limit=21.0,
-    saturation_effort=44.4,
-    stiffness=stiffness_mujoco,
-    damping=damping_mujoco,
-    armature=armature_mujoco,
-    friction_static=friction_static_mujoco,
-    activation_vel=0.1,
-    friction_dynamic=friction_dynamic_mujoco,
-)
-
-ALIENGO_THIGH_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
-    joint_names_expr=[".*_thigh_joint"],
-    effort_limit=44.4,
-    velocity_limit=21.0,
-    saturation_effort=44.4,
-    stiffness=stiffness_mujoco,
-    damping=damping_mujoco,
-    armature=armature_mujoco,
-    friction_static=friction_static_mujoco,
-    activation_vel=0.1,
-    friction_dynamic=friction_dynamic_mujoco,
-)
-
-ALIENGO_CALF_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
-    joint_names_expr=[".*_calf_joint"],
-    effort_limit=44.4,
-    velocity_limit=21.0,
-    saturation_effort=44.4,
-    stiffness=stiffness_mujoco,
-    damping=damping_mujoco,
-    armature=armature_mujoco,
-    friction_static=friction_static_mujoco,
-    activation_vel=0.1,
-    friction_dynamic=friction_dynamic_mujoco,
-)
-
-ALIENGO_ARM_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
-    joint_names_expr=["arm_joint.*"],
-    effort_limit=30.0, #TODO, the joint2 has 60 as limits
-    velocity_limit=3.1415,
-    saturation_effort=10.0,
-    stiffness=50.0,
-    damping=5.0,
-    armature=0.01,
-    friction_static=0.1,
-    activation_vel=0.1,
-    friction_dynamic=0.1,
-)"""
-
-ALIENGO_HIP_ACTUATOR_CFG = DelayedPDActuatorCfg(
-    joint_names_expr=[".*_hip_joint"],
-    effort_limit=44.4,
-    stiffness=stiffness_mujoco,
-    damping=damping_mujoco,
-    min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
-    max_delay=2,  # physics time steps (max: 2.0*2=4.0ms)
-)
-
-ALIENGO_THIGH_ACTUATOR_CFG = DelayedPDActuatorCfg(
-    joint_names_expr=[".*_thigh_joint"],
-    effort_limit=44.4,
-    stiffness=stiffness_mujoco,
-    damping=damping_mujoco,
-    min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
-    max_delay=2,  # physics time steps (max: 2.0*2=4.0ms)
-)
-
-ALIENGO_CALF_ACTUATOR_CFG = DelayedPDActuatorCfg(
-    joint_names_expr=[".*_calf_joint"],
-    effort_limit=44.4,
-    stiffness=stiffness_mujoco,
-    damping=damping_mujoco,
-    min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
-    max_delay=2,  # physics time steps (max: 2.0*2=4.0ms)
-)
-
-
-ALIENGO_ARM_ACTUATOR_CFG = DelayedPDActuatorCfg(
-    joint_names_expr=["arm_joint.*"],
-    effort_limit=30.0,
-    stiffness=50.0,
-    damping=5.0,
-    min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
-    max_delay=2,  # physics time steps (max: 2.0*2=4.0ms)
-)
-
-ALIENGO_CFG = ArticulationCfg(
-    prim_path=None,
-    spawn=sim_utils.UsdFileCfg(
-        usd_path=f"{ISAAC_ASSET_DIR}/aliengo_z1_nogripper_arm_nocontact.usd",
-        activate_contact_sensors=True,
-        rigid_props=sim_utils.RigidBodyPropertiesCfg(
-            disable_gravity=False,
-            retain_accelerations=False,
-            linear_damping=0.0,
-            angular_damping=0.0,
-            max_linear_velocity=1000.0,
-            max_angular_velocity=1000.0,
-            max_depenetration_velocity=1.0,
-        ),
-        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
-            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
-        ),
-    ),
-    init_state=ArticulationCfg.InitialStateCfg(
-        pos=(0.0, 0.0, 0.4),
-        joint_pos={
-            ".*L_hip_joint": 0.0,
-            ".*R_hip_joint": 0.0,
-            ".*_thigh_joint": 0.9,
-            ".*_calf_joint": -1.8,
-            "arm_joint1": 0.0,
-            "arm_joint2": 0.0,
-            "arm_joint3": 0.0,
-            "arm_joint4": 0.0,
-            "arm_joint5": 0.0,
-            "arm_joint6": 0.0,
-        },
-        joint_vel={".*": 0.0},
-    ),
-
-    actuators={"hip": ALIENGO_HIP_ACTUATOR_CFG, "thigh": ALIENGO_THIGH_ACTUATOR_CFG, "calf": ALIENGO_CALF_ACTUATOR_CFG,
-               "arm": ALIENGO_ARM_ACTUATOR_CFG},
-    soft_joint_pos_limit_factor=0.95,
-)
diff --git a/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env.py b/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env.py
index 6906379..ac72542 100644
--- a/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env.py
+++ b/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env.py
@@ -512,7 +512,8 @@ class LocomotionEnv(DirectRLEnv):
         # Reset robot state
         joint_pos = self._robot.data.default_joint_pos[env_ids]
         joint_pos[:, self._ids_only_arms_joints_order] += torch.zeros_like(joint_pos[:, self._ids_only_arms_joints_order]).uniform_(-3.14, 3.14)
-        # we need to project them inside the robots limits!
+        joint_pos[:, self._ids_only_legs_joints_order] += torch.zeros_like(joint_pos[:, self._ids_only_legs_joints_order]).uniform_(-0.2, 0.2)
+        # we need to project them inside the robots limits (only arms!)
         joints_limits = self._robot.data.default_joint_pos_limits
         joints_arm_limits = joints_limits[:,self._ids_only_arms_joints_order]
         joint_pos[:, self._ids_only_arms_joints_order] = torch.clamp(joint_pos[:, self._ids_only_arms_joints_order], joints_arm_limits[0,:,0], joints_arm_limits[0,:,1])
diff --git a/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env_cfg.py b/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env_cfg.py
index e2f4a77..77cbec8 100644
--- a/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env_cfg.py
+++ b/source/trash_collection_isaaclab/trash_collection_isaaclab/tasks/locomotion/locomotion_env_cfg.py
@@ -69,9 +69,9 @@ class EventCfg:
         mode="reset",
         params={
             "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
-            "friction_distribution_params": (0.2, 2.0),
-            "armature_distribution_params": (0.0, 1.0),
-            "operation": "abs",
+            "friction_distribution_params": (0.1, 2.0),
+            "armature_distribution_params": (1.0, 2.0),
+            "operation": "scale",
             "distribution": "uniform",
         },
     )
@@ -286,7 +286,7 @@ class AliengoFlatEnvCfg(DirectRLEnvCfg):
     action_smoothness_reward_scale = -0.001
 
     # Feet reward scale
-    feet_air_time_reward_scale = 0.5
+    feet_air_time_reward_scale = 0.5 * 0.0
     feet_height_clearance_reward_scale = 0.25  
     feet_height_clearance_mujoco_reward_scale = 0.25 * 0.0
     feet_slide_reward_scale = -0.25 * 0.0
