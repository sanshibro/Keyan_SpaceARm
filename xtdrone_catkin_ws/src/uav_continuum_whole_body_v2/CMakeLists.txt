cmake_minimum_required(VERSION 3.0.2)
project(uav_continuum_whole_body_v2)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  trajectory_msgs
  visualization_msgs
)

add_message_files(FILES
  WholeBodyTrajectory.msg
)

# The workstation has a newer user-local setuptools paired with Ubuntu's old
# importlib_metadata. Catkin must use the coherent ROS/system Python toolchain.
set(ENV{PYTHONNOUSERSITE} 1)
catkin_python_setup()

generate_messages(DEPENDENCIES
  geometry_msgs
  std_msgs
  std_srvs
)

catkin_package(CATKIN_DEPENDS
  geometry_msgs
  message_runtime
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  trajectory_msgs
  visualization_msgs
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_core.py)
endif()

catkin_install_python(PROGRAMS
  scripts/rolling_whole_body_planner_node.py
  scripts/whole_body_trajectory_executor_node.py
  scripts/generate_unified_scenario.py
  scripts/run_offline_demo.py
  scripts/whole_body_metrics_logger_node.py
  scripts/whole_body_visualization_node.py
  scripts/check_v2_live.py
  scripts/analyze_v2_run.py
  scripts/safe_takeoff_hold_node.py
  scripts/calibrate_px4_tracking.py
  scripts/calibrate_px4_stopping.py
  scripts/slam_map_snapshot_node.py
  scripts/register_depth_cloud_node.py
  scripts/anchored_estimator_camera_pose_node.py
  scripts/terminal_yaw_map_scan_node.py
  scripts/independent_map_survey_node.py
  scripts/spawn_comparison_scene_node.py
  scripts/check_method_comparison_live.py
  scripts/slam_reference_planner_node.py
  scripts/comparison_method_planner_node.py
  scripts/generate_comparison_scenarios.py
  scripts/run_offline_method_comparison.py
  scripts/run_method_comparison_pilot.py
  scripts/run_closed_loop_method_comparison.py
  scripts/run_formal_comparison_batch.py
  scripts/aggregate_method_comparison.py
  scripts/audit_slam_map.py
  scripts/fuse_slam_maps.py
  scripts/generate_independent_survey_manifest.py
  scripts/certify_independent_survey.py
  scripts/audit_v21_existing_runs.py
  scripts/summarize_v21_e2.py
  scripts/analyze_v21_dynamics.py
  scripts/analyze_core_ablation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/run_scenario_3c_acceptance.sh
  scripts/run_method_comparison_closed_loop.sh
  scripts/record_and_audit_slam_map.sh
  scripts/run_independent_map_survey.sh
  scripts/run_formal_comparison_pipeline.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch generated
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
