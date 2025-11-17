# Overview

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




## Installation

If you want only to deploy a trained policy on your robot, go directly [at the bottom of the readme](https://github.com/iit-DLSLab/basic-locomotion-dls-isaaclab/tree/main?tab=readme-ov-file#run-sim-to-sim-and-sim-to-real).

1. Install Isaac Lab by following the [installation guide](https://github.com/isaac-sim/IsaacLab). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

2. Install git for very large file
```bash
sudo apt install git-lfs
```

3. Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)


4. Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/trash_collection_isaaclab
```



### Run a train/play in IsaacLab

- To train:

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Flat --num_envs=4096 --headless
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=4096 --headless
python scripts/rsl_rl/train.py --task=Manipulation-Aliengo-Rough-Blind --num_envs=4096 --headless
```

### Run Hyperparameter Search

```bash
echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | python3 (TERMINAL 1)
```

```bash
python3 ../basic_locomotion_dls_isaaclab/exts/basic_locomotion_dls_isaaclab/basic_locomotion_dls_isaaclab/hyperparameter_tuning/tuner.py --run_mode local --cfg_file ../basic_locomotion_dls_isaaclab/exts/basic_locomotion_dls_isaaclab/basic_locomotion_dls_isaaclab/hyperparameter_tuning/locomotion_aliengo_cfg.py --cfg_class LocomotionAliengoFlatTuner (TERMINAL 2)
```


### Convert XML to USD
We use models saved [here](https://github.com/iit-DLSLab/trash-collection-isaaclab/tree/main/deploy/mujoco/models).

```bash
./isaaclab.sh -p scripts/tools/convert_mjcf.py   ../basic_locomotion_dls_isaaclab/scripts/sim_to_sim_mujoco/gym-quadruped/gym_quadruped/robot_model/aliengo/aliengo.xml   ../aliengo.usd   --import-sites   --make-instanceable
```

Remember to set in the application above, "set as default prim" to the root of the robot. Furthermore, for now, add the following lines in the xml of your robots to make the feet seen as body

```bash
<body name="FL_foot" pos="0 0 -0.25">
    <!-- FL_foot only collision -->
    <geom name="FL" class="collision" size="0.0265" pos="0 0 0" />
</body>
```


### Run Sim-to-Sim and Sim-to-Real

1. install [miniforge](https://github.com/conda-forge/miniforge/releases) (x86_64 or arm64 depending on your platform)

2. create an conda environment using the file in the folder [installation](https://github.com/iit-DLSLab/trash-collection-isaaclab/tree/main/deploy/installation):


```bash
conda env create -f mamba_environment_ros1.yaml
conda activate trash_collection_isaaclab_ros2_env
```

or using docker
```bash
docker build -t trash_collection_isaaclab_image .
```

putting in your .bashrc the following alias
```bash
alias trash_collection_isaaclab_docker='
if [ ! "$(docker ps -a -q -f name=trash_collection_isaaclab_container)" ]; then
   xhost + && docker run -it --rm -v absolute_path_to_this_repo:/home/ -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/input/ -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e QT_X11_NO_MITSHM=1 --gpus all --net host --cap-add=sys_nice --name trash_collection_isaaclab_container trash_collection_isaaclab_image; \
else
   docker exec -it trash_collection_isaaclab_container bash; \
fi'
```

3. Then you can 

```bash
## Sim-to-Sim
python3 deploy/play_mujoco.py

## Sim-to-Real with ROS2
cd deploy/ros2_ws
colcon build
python3 deploy/play_ros2.py 
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick)
```
