## Installation Deploy unsing Conda

1. install [miniforge](https://github.com/conda-forge/miniforge/releases) (x86_64 or arm64 depending on your platform)

2. create an conda environment using the file in the folder [installation](https://github.com/iit-DLSLab/trash-collection-isaaclab/tree/main/deploy/installation):


```bash
conda env create -f mamba_environment_ros2.yaml
conda activate trash_collection_isaaclab_ros2_env
```

## Installation Deploy unsing Docker 
1. install docker and run
```bash
docker build -t trash_collection_isaaclab_image .
```

2. put in your .bashrc the following alias
```bash
alias trash_collection_isaaclab_docker='
if [ ! "$(docker ps -a -q -f name=trash_collection_isaaclab_container)" ]; then
   xhost + && docker run -it --rm -v absolute_path_to_this_repo:/home/ -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/input/ -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e QT_X11_NO_MITSHM=1 --gpus all --net host --cap-add=sys_nice --name trash_collection_isaaclab_container trash_collection_isaaclab_image; \
else
   docker exec -it trash_collection_isaaclab_container bash; \
fi'
```

3. or use docker compose

```bash
alias trash_collection_isaaclab_docker_compose='xhost + && cd absolute_path_to_this_repo/deploy/installation && docker compose -f docker-compose.yaml run trash-docker bash'
```

Remember to update absolute_path_to_this_repo !!

## Run Sim-to-Sim 

```bash
## Sim-to-Sim sequential
python3 deploy/play_mujoco.py

## Sim-to-Sim with ROS2
source deploy/ros2_localhost_connect.sh (TERMINAL 1)
python3 deploy/run_controller_ros2.py (TERMINAL 1)

source deploy/ros2_localhost_connect.sh (TERMINAL 2)
python3 deploy/run_simulator_ros2.py (TERMINAL 2)

source deploy/ros2_localhost_connect.sh (TERMINAL 3)
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 3)
```

## Run Sim-to-Real

```bash
## Sim-to-Real with ROS2
source deploy/dls2_connect.sh (TERMINAL 1)
python3 deploy/run_controller_ros2.py (TERMINAL 1)

source deploy/dls2_connect.sh (TERMINAL 2)
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 2)
```
