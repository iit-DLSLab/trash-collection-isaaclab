## Installation Deploy unsing Conda

1. install [miniforge](https://github.com/conda-forge/miniforge/releases) (x86_64 or arm64 depending on your platform)

2. create an conda environment using the file in the folder [installation](https://github.com/iit-DLSLab/trash-collection-isaaclab/tree/main/deploy/installation):


```bash
conda env create -f mamba_environment_ros1.yaml
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
Remember to update absolute_path_to_this_repo !!

## Run Sim-to-Sim 

```bash
## Sim-to-Sim sequential
python3 deploy/play_mujoco.py

## Sim-to-Sim with ROS2
cd deploy/ros2_ws
colcon build
source install/setup.bash
python3 deploy/run_controller_ros2.py  
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick)
python3 deploy/run_simulator_ros2.py
```

## Run Sim-to-Real

```bash
## Sim-to-Real with ROS2
cd deploy/ros2_ws
colcon build
source install/setup.bash
./dls2_connect.sh
python3 deploy/run_controller_ros2.py
```
If you want a joystick, on a second terminal enter in the docker and then:

```bash
cd deploy/
source ros2_ws/install/setup.bash
./dls2_connect.sh
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox'
```
