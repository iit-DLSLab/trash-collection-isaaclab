## Overview

This repository is about trash collection using RL on quadruped robots, with sim-to-sim and sim-to-real scripts. It trains in sequence two different policies, one for locomotion and a second one for manipulation, able to reach desired end effector poses.

Features:
- Locomotion policy able to adjust pose and carry a manipulator
- Manipulation policy able to reach desired end effector goal while coordinating the pose of the quadruped
- Sim-to-Sim in [Mujoco](https://github.com/google-deepmind/mujoco)
- Sim-to-Real in ROS1 and ROS2

A list of robots available and envs are described below:

| Robot Model         | Environment Name (ID)                                      |
|---------------------|------------------------------------------------------------|
| [Aliengo](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/aliengo) | Locomotion-Aliengo-Flat, Locomotion-Aliengo-Rough-Blind, Locomotion-Aliengo-Rough-Vision |
| [Arm with Z1](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/aliengo) | Manipulation-Aliengo-Flat, Manipulation-Aliengo-Rough-Blind, Manipulation-Aliengo-Rough-Vision |




## Installation and Runs

If you want only to deploy a trained policy on your robot, continue on [README_DEPLOY](https://github.com/iit-DLSLab/trash-collection-isaaclab/blob/main/README_deploy.md) otherwise on [README_TRAIN](https://github.com/iit-DLSLab/trash-collection-isaaclab/blob/main/README_train.md).

