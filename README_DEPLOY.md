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
