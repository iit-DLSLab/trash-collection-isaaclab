from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib.keyboard_teleop import keycodes

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")

@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(dir_path+"/mujoco/models/z1/scene_floating.xml")
    data = mujoco.MjData(model)

    # Joints we wish to control.
    # fmt: off
    joint_names = [
        "basepitch", "basez", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
    ]
    # fmt: on
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    #actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_cost = np.zeros((model.nv,))
    posture_cost[2:] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[0:2] = 100
    damping_task = mink.DampingTask(model, immobile_base_cost)

    tasks = [
        end_effector_task,
        posture_task,
    ]

    # Enable collision avoidance between the following geoms.
    # left hand - table, right hand - table
    # left hand - left thigh, right hand - right thigh
    collision_pairs = [
        (["link02_geom2"], ["trunk"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.005,
        collision_detection_distance=0.15,
    )

    limits = [
        mink.ConfigurationLimit(model),
        collision_avoidance_limit,
    ]

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                if key_callback.fix_base:
                    vel = mink.solve_ik(
                        configuration,
                        [*tasks, damping_task],
                        rate.dt,
                        solver,
                        damping=1e-3,
                    )
                else:
                    vel = mink.solve_ik(
                        configuration, tasks, rate.dt, solver, damping=1e-3
                    )
                configuration.integrate_inplace(vel, rate.dt)

                # Exit condition.
                err = end_effector_task.compute_error(configuration)
                pos_achieved = bool(np.linalg.norm(err[:3]) <= pos_threshold)
                ori_achieved = bool(np.linalg.norm(err[3:]) <= ori_threshold)
                if pos_achieved and ori_achieved:
                    break

            if not key_callback.pause:
                #data.ctrl[actuator_ids] = configuration.q[dof_ids]
                #mujoco.mj_step(model, data)
                data.qpos[dof_ids] = configuration.q[dof_ids]
                mujoco.mj_kinematics(model, data)
                
            else:
                mujoco.mj_kinematics(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
