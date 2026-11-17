cmake_minimum_required(VERSION 3.0.2)
project(xtdrone_ego_tools)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/model_state_to_odom.py
  scripts/pose_to_gazebo_model.py
  scripts/gazebo_truth_bridge.py
  scripts/set_px4_sitl_params.py
  scripts/check_ego_inputs.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
