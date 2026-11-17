cmake_minimum_required(VERSION 3.0.2)
project(uav_continuum_opportunity_planner)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  traj_utils
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/arm_kinematics_node.py
  scripts/arm_shape_sim_node.py
  scripts/export_world_boxes.py
  scripts/export_gazebo_scene_cloud.py
  scripts/obstacle_geometry_node.py
  scripts/global_path_node.py
  scripts/ego_reference_bridge_node.py
  scripts/whole_body_planner_node.py
  scripts/joint_planner_node.py
  scripts/plan_executor_node.py
  scripts/px4_waypoint_executor_node.py
  scripts/reachability_node.py
  scripts/spawn_task_scene_node.py
  scripts/opportunity_visualization_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
