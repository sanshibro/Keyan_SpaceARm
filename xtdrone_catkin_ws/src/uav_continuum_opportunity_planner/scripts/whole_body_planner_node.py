#!/usr/bin/env python3
import heapq
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
import tf.transformations as tft
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from visualization_msgs.msg import Marker, MarkerArray

from uav_continuum_opportunity_planner.common import (
    VoxelOccupancy,
    arm_points,
    color,
    load_arm_params,
    load_body_keepout_boxes,
    load_boxes,
    load_target,
    marker_header,
    mount_matrix,
    point_distance,
    point_msg,
    pose_to_matrix,
    transform_point,
    world_arm_points,
)

PLANNER_VERSION = "whole_body_v20260704_dynamics_aware_sequence_v1"


def param_bool(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "on"]
    return bool(value)


def param_xyz(name, default):
    value = rospy.get_param(name, default)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        parts = [float(item) for item in value]
    if len(parts) != 3:
        raise ValueError("%s must contain exactly three values" % name)
    return tuple(parts)


class WholeBodyPlannerNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.scenario_name = rospy.get_param("/scenario/name", "unknown")
        self.arm_params = load_arm_params()
        self.boxes = load_boxes()
        self.body_keepout_boxes = load_body_keepout_boxes()
        self.target = load_target()

        vehicle = rospy.get_param("/vehicle", {})
        morphology = rospy.get_param("/morphology", {})
        self.body_radius = float(vehicle.get("body_radius", 0.34))
        self.body_height = float(vehicle.get("body_height", self.body_radius * 2.0))
        self.body_clearance = float(vehicle.get("body_clearance", 0.04))
        self.arm_clearance = float(vehicle.get("arm_clearance", 0.025))
        self.compact_theta = float(morphology.get("compact_theta_deg", 100.0))
        self.compact_phi = float(morphology.get("compact_phi_deg", 0.0))
        self.max_delta_theta = float(morphology.get("max_delta_theta_deg", 15.0))
        self.max_delta_phi = float(morphology.get("max_delta_phi_deg", 25.0))
        self.target_tolerance = float(rospy.get_param("/reachability/target_tolerance", 0.045))
        self.enable_whole_body_refinement = param_bool("~enable_whole_body_refinement", True)
        self.refine_lateral_step = float(rospy.get_param("~refine_lateral_step", 0.16))
        self.refine_z_step = float(rospy.get_param("~refine_z_step", 0.05))
        self.refine_yaw_step_deg = float(rospy.get_param("~refine_yaw_step_deg", 12.0))
        self.refine_clearance_target = float(rospy.get_param("~refine_clearance_target", 0.08))
        self.refine_stride = max(1, int(rospy.get_param("~refine_stride", 4)))
        self.refine_max_points = max(4, int(rospy.get_param("~refine_max_points", 36)))
        self.refine_compact_weight = float(rospy.get_param("~refine_compact_weight", 0.018))
        self.refine_com_weight = float(rospy.get_param("~refine_com_weight", 2.4))
        self.refine_lateral_extent_weight = float(rospy.get_param("~refine_lateral_extent_weight", 0.8))
        self.refine_clearance_weight = float(rospy.get_param("~refine_clearance_weight", 10.0))
        self.refine_min_arm_margin = float(rospy.get_param("~refine_min_arm_margin", 0.03))
        self.morphology_home_hold_weight = float(rospy.get_param("~morphology_home_hold_weight", 0.08))
        self.morphology_activation_margin = float(rospy.get_param("~morphology_activation_margin", 0.08))
        self.morphology_activation_band = max(1e-3, float(rospy.get_param("~morphology_activation_band", 0.08)))
        self.morphology_min_active_theta = float(rospy.get_param("~morphology_min_active_theta", 35.0))
        self.passage_morphology_gate_enabled = param_bool("~passage_morphology_gate_enabled", True)
        self.passage_compact_arm_margin_buffer = float(rospy.get_param("~passage_compact_arm_margin_buffer", 0.015))
        self.passage_compact_body_margin_buffer = float(rospy.get_param("~passage_compact_body_margin_buffer", 0.020))
        self.passage_compact_shape_weight = float(rospy.get_param("~passage_compact_shape_weight", 42.0))
        self.passage_transition_shape_weight = float(rospy.get_param("~passage_transition_shape_weight", 18.0))
        self.morphology_release_start_fraction = float(rospy.get_param("~morphology_release_start_fraction", 0.62))
        self.morphology_release_target_distance = float(rospy.get_param("~morphology_release_target_distance", 3.2))
        self.morphology_q_ref_weight = float(rospy.get_param("~morphology_q_ref_weight", 0.018))
        self.morphology_stability_shape_weight = float(rospy.get_param("~morphology_stability_shape_weight", 0.020))
        self.morphology_lateral_extent_weight = float(rospy.get_param("~morphology_lateral_extent_weight", 1.8))
        self.morphology_release_transition_weight = float(rospy.get_param("~morphology_release_transition_weight", 8.0))
        self.refine_shape_theta_stride = max(1, int(rospy.get_param("~refine_shape_theta_stride", 3)))
        self.refine_shape_phi_stride = max(1, int(rospy.get_param("~refine_shape_phi_stride", 3)))
        self.refine_time_budget = float(rospy.get_param("~refine_time_budget", 15.0))
        self.refine_wp_time_budget = float(rospy.get_param("~refine_wp_time_budget", 0.8))
        self.refine_valid_candidate_target = max(1, int(rospy.get_param("~refine_valid_candidate_target", 80)))
        self.refine_shape_candidates_per_pose = max(1, int(rospy.get_param("~refine_shape_candidates_per_pose", 80)))
        self.reach_search_time_budget = float(rospy.get_param("~reach_search_time_budget", 3.0))
        self.reach_search_max_states = max(50, int(rospy.get_param("~reach_search_max_states", 800)))
        self.marker_draw_all_arm_waypoints = param_bool("~marker_draw_all_arm_waypoints", True)
        self.marker_arm_visual_spacing = max(0.03, float(rospy.get_param("~marker_arm_visual_spacing", 0.15)))
        self.marker_arm_visual_theta_step = max(1.0, float(rospy.get_param("~marker_arm_visual_theta_step", 7.5)))
        self.marker_arm_visual_phi_step = max(1.0, float(rospy.get_param("~marker_arm_visual_phi_step", 12.5)))
        self.marker_waypoint_label_stride = max(1, int(rospy.get_param("~marker_waypoint_label_stride", 1)))
        self.freeze_plan_when_ready = param_bool("~freeze_plan_when_ready", True)
        self.enable_reference_safety_repair = param_bool("~enable_reference_safety_repair", False)
        self.enable_global_repair_fallback = param_bool("~enable_global_repair_fallback", False)
        self.failure_hold_time = float(rospy.get_param("~failure_hold_time", 8.0))
        self.refine_first_feasible = param_bool("~refine_first_feasible", False)
        self.last_reach_search_info = {}

        global_cfg = rospy.get_param("/global_path", {})
        base_resolution = float(global_cfg.get("resolution", 0.20))
        self.skeleton_xy_resolution = max(
            0.05,
            float(rospy.get_param("~skeleton_xy_resolution", global_cfg.get("xy_resolution", base_resolution))),
        )
        self.skeleton_z_resolution = max(
            0.03,
            float(rospy.get_param("~skeleton_z_resolution", global_cfg.get("z_resolution", base_resolution))),
        )
        self.skeleton_z_min = float(rospy.get_param("~skeleton_z_min", global_cfg.get("z_min", 0.90)))
        self.skeleton_z_max = float(rospy.get_param("~skeleton_z_max", global_cfg.get("z_max", 2.05)))
        self.skeleton_xy_padding = float(rospy.get_param("~skeleton_xy_padding", global_cfg.get("xy_padding", 2.5)))
        self.skeleton_max_expansions = max(
            1000,
            int(rospy.get_param("~skeleton_max_expansions", global_cfg.get("max_expansions", 160000))),
        )
        self.skeleton_heuristic_weight = float(
            rospy.get_param("~skeleton_heuristic_weight", global_cfg.get("heuristic_weight", 1.8))
        )
        self.skeleton_path_step = max(0.05, float(rospy.get_param("~skeleton_path_step", 0.20)))
        self.sequence_optimizer_enabled = param_bool("~sequence_optimizer_enabled", True)
        self.robust_margin_enabled = param_bool("~robust_margin_enabled", True)
        self.local_repair_enabled = param_bool("~local_repair_enabled", True)
        self.time_allocation_refine_enabled = param_bool("~time_allocation_refine_enabled", True)
        self.swept_validation_enabled = param_bool("~swept_validation_enabled", True)

        self.m_uav = float(rospy.get_param("~m_uav", 1.50))
        self.m_arm_total = float(rospy.get_param("~m_arm_total", 0.75))
        self.m_arm_base = float(rospy.get_param("~m_arm_base", 0.30))
        self.m_flexible_section = float(rospy.get_param("~m_flexible_section", 0.25))
        self.m_tool = float(rospy.get_param("~m_tool", 0.20))
        self.m_system_total = max(self.m_uav + self.m_arm_total, 1e-6)
        self.preferred_com_offset = float(rospy.get_param("~preferred_com_offset", 0.05))
        self.hard_com_limit = float(rospy.get_param("~hard_com_limit", 0.08))
        self.tau_limit = float(rospy.get_param("~tau_limit", self.m_system_total * 9.81 * self.hard_com_limit))
        self.com_weight = float(rospy.get_param("~com_weight", 14.0))
        self.torque_weight = float(rospy.get_param("~torque_weight", 1.2))

        self.localization_margin = float(rospy.get_param("~localization_margin", 0.03))
        self.uav_tracking_margin = float(rospy.get_param("~uav_tracking_margin", 0.03))
        self.arm_model_margin = float(rospy.get_param("~arm_model_margin", 0.03))
        self.arm_tracking_margin = float(rospy.get_param("~arm_tracking_margin", 0.03))
        self.unmodeled_airflow_margin = float(rospy.get_param("~unmodeled_airflow_margin", 0.02))
        self.required_body_margin = float(
            rospy.get_param(
                "~required_body_margin",
                max(0.05, self.localization_margin + self.uav_tracking_margin),
            )
        )
        self.required_arm_margin = float(
            rospy.get_param(
                "~required_arm_margin",
                max(0.05, self.arm_model_margin + self.arm_tracking_margin + self.unmodeled_airflow_margin),
            )
        )
        self.standoff_body_margin_target = float(
            rospy.get_param("~standoff_body_margin_target", max(self.required_body_margin + 0.08, 0.14))
        )
        self.standoff_arm_margin_target = float(
            rospy.get_param("~standoff_arm_margin_target", max(self.required_arm_margin + 0.06, 0.10))
        )
        self.standoff_approach_distance = float(rospy.get_param("~standoff_approach_distance", 0.75))
        self.standoff_approach_z_lift = float(rospy.get_param("~standoff_approach_z_lift", 0.18))
        self.standoff_candidate_score_count = max(1, int(rospy.get_param("~standoff_candidate_score_count", 160)))
        self.standoff_approach_rerank_count = max(1, int(rospy.get_param("~standoff_approach_rerank_count", 12)))
        self.standoff_selection_time_budget = max(0.0, float(rospy.get_param("~standoff_selection_time_budget", 4.0)))
        self.swept_repair_body_margin_buffer = float(rospy.get_param("~swept_repair_body_margin_buffer", 0.035))
        self.swept_repair_arm_margin_buffer = float(rospy.get_param("~swept_repair_arm_margin_buffer", 0.020))
        self.sequence_risk_body_margin = float(
            rospy.get_param("~sequence_risk_body_margin", max(self.refine_clearance_target, self.required_body_margin + 0.02))
        )
        self.sequence_risk_arm_margin = float(
            rospy.get_param("~sequence_risk_arm_margin", max(self.refine_clearance_target, self.required_arm_margin + 0.045))
        )
        self.edge_body_margin_target = float(
            rospy.get_param(
                "~edge_body_margin_target",
                max(self.required_body_margin + 0.02, self.sequence_risk_body_margin),
            )
        )
        self.edge_arm_margin_target = float(
            rospy.get_param(
                "~edge_arm_margin_target",
                max(self.required_arm_margin + 0.015, self.sequence_risk_arm_margin),
            )
        )

        self.v_max = float(rospy.get_param("~v_max", 0.60))
        self.a_max = float(rospy.get_param("~a_max", 1.00))
        self.j_max = float(rospy.get_param("~j_max", 3.00))
        self.yaw_rate_max = math.radians(float(rospy.get_param("~yaw_rate_max_deg", 45.0)))
        self.yaw_acc_max = math.radians(float(rospy.get_param("~yaw_acc_max_deg", 90.0)))
        self.tilt_max = math.radians(float(rospy.get_param("~tilt_max_deg", 15.0)))
        self.thrust_min = float(rospy.get_param("~thrust_min", 0.50 * self.m_system_total * 9.81))
        self.thrust_max = float(rospy.get_param("~thrust_max", 1.60 * self.m_system_total * 9.81))
        self.nominal_speed = max(0.05, float(rospy.get_param("~nominal_speed", 0.35)))
        self.segment_dt_min = max(0.05, float(rospy.get_param("~segment_dt_min", 0.20)))
        self.segment_dt_max = max(self.segment_dt_min, float(rospy.get_param("~segment_dt_max", 2.50)))
        self.time_refine_max_scale = max(1.0, float(rospy.get_param("~time_refine_max_scale", 6.0)))

        self.theta_dot_max = float(rospy.get_param("~theta_dot_max_deg_s", 50.0))
        self.phi_dot_max = float(rospy.get_param("~phi_dot_max_deg_s", 60.0))
        self.theta_acc_max = float(rospy.get_param("~theta_acc_max_deg_s2", 100.0))
        self.phi_acc_max = float(rospy.get_param("~phi_acc_max_deg_s2", 120.0))
        self.bending_weight = float(rospy.get_param("~bending_weight", 0.015))

        self.swept_translation_step = max(0.01, float(rospy.get_param("~swept_translation_step", 0.06)))
        self.swept_theta_step = max(1.0, float(rospy.get_param("~swept_theta_step_deg", 8.0)))
        self.swept_phi_step = max(1.0, float(rospy.get_param("~swept_phi_step_deg", 12.0)))
        self.sequence_transition_neighbors = max(4, int(rospy.get_param("~sequence_transition_neighbors", 24)))
        self.sequence_beam_width = max(4, int(rospy.get_param("~sequence_beam_width", 24)))
        self.sequence_edge_eval_neighbors = max(2, int(rospy.get_param("~sequence_edge_eval_neighbors", 8)))
        self.sequence_edge_collision_lazy = param_bool("~sequence_edge_collision_lazy", True)
        self.sequence_refine_max_points = max(6, int(rospy.get_param("~sequence_refine_max_points", 36)))
        self.sequence_dp_candidates_per_wp = max(4, int(rospy.get_param("~sequence_dp_candidates_per_wp", 10)))
        self.sequence_candidate_budget_fraction = min(0.90, max(0.25, float(rospy.get_param("~sequence_candidate_budget_fraction", 0.65))))
        self.sequence_dp_min_time_budget = max(0.0, float(rospy.get_param("~sequence_dp_min_time_budget", 6.0)))
        self.sequence_candidates_per_wp = max(6, int(rospy.get_param("~sequence_candidates_per_wp", 32)))
        self.sequence_min_candidates_per_wp = max(1, int(rospy.get_param("~sequence_min_candidates_per_wp", min(12, self.sequence_candidates_per_wp))))
        self.sequence_max_pose_tests_per_wp = max(8, int(rospy.get_param("~sequence_max_pose_tests_per_wp", 72)))
        self.sequence_max_body_poses_per_wp = max(2, int(rospy.get_param("~sequence_max_body_poses_per_wp", 14)))
        self.sequence_max_shape_tests_per_wp = max(20, int(rospy.get_param("~sequence_max_shape_tests_per_wp", 520)))
        self.local_repair_max_attempts = max(0, int(rospy.get_param("~local_repair_max_attempts", 1)))
        self.edge_repair_enabled = param_bool("~edge_repair_enabled", True)
        self.edge_repair_max_pairs = max(1, int(rospy.get_param("~edge_repair_max_pairs", 8)))
        self.edge_repair_max_ladders = max(1, int(rospy.get_param("~edge_repair_max_ladders", 8)))
        self.edge_repair_candidates_per_knot = max(2, int(rospy.get_param("~edge_repair_candidates_per_knot", 4)))
        self.edge_repair_max_results = max(1, int(rospy.get_param("~edge_repair_max_results", 4)))
        self.edge_repair_max_knots = max(1, int(rospy.get_param("~edge_repair_max_knots", 2)))
        self.edge_repair_time_budget = max(0.0, float(rospy.get_param("~edge_repair_time_budget", 1.2)))

        self.use_static_cloud = bool(rospy.get_param("~use_static_cloud", True))
        self.static_cloud_in_optimization = param_bool("~static_cloud_in_optimization", True)
        self.static_cloud_in_validation = param_bool("~static_cloud_in_validation", True)
        self.cloud_collision_enabled = self.use_static_cloud
        self.require_static_cloud = bool(rospy.get_param("~require_static_cloud", False))
        self.static_cloud_topic = rospy.get_param("~static_cloud_topic", "/world/static_scene_cloud")
        self.static_cloud_search_radius = float(rospy.get_param("~static_cloud_search_radius", 0.45))
        self.static_cloud_query_padding = float(rospy.get_param("~static_cloud_query_padding", 0.05))
        self.static_cloud_stride = max(1, int(rospy.get_param("~static_cloud_stride", 1)))
        static_resolution = float(rospy.get_param("~static_cloud_resolution", 0.10))
        static_inflation = float(rospy.get_param("~static_cloud_inflation", 0.05))
        static_max_points = int(rospy.get_param("~static_cloud_max_points", 250000))
        static_distance_field = param_bool("~static_cloud_distance_field", True)
        static_distance_field_padding = float(rospy.get_param("~static_cloud_distance_field_padding", 0.75))
        static_distance_field_max_cells = int(rospy.get_param("~static_cloud_distance_field_max_cells", 8000000))
        static_distance_field_min = param_xyz("~static_cloud_distance_field_min", [-1.8, -13.2, 0.0])
        static_distance_field_max = param_xyz("~static_cloud_distance_field_max", [3.4, 7.8, 2.3])
        self.static_occupancy = VoxelOccupancy(
            static_resolution,
            static_inflation,
            static_max_points,
            static_distance_field,
            static_distance_field_padding,
            static_distance_field_max_cells,
            static_distance_field_min,
            static_distance_field_max,
        )
        self.static_cloud_loaded = False

        self.current_pose = None
        self.current_theta_phi = [0.0, 0.0]
        self.global_path = None
        self.global_path_stamp = rospy.Time(0)
        self.global_path_cb_count = 0
        self.global_path_array_cb_count = 0
        self.reference_path = None
        self.reference_cb_count = 0
        self.reference_array_cb_count = 0
        self.selected_goal = None
        self.reach_candidates = []
        self.last_plan_time = rospy.Time(0)
        self.plan_period = float(rospy.get_param("~plan_period", 2.0))
        self.last_marker_arm_count = 0
        self.last_interpolated_arm_marker_count = 0
        self.last_low_clearance_marker_count = 0
        self.last_waypoint_label_count = 0
        self.frozen_plan_result = None
        self.frozen_plan_status = None
        self.input_wait_timeout = float(rospy.get_param("~input_wait_timeout", 2.0))
        self.input_refresh_attempts = 0
        self.last_global_path_wait_ok = False
        self.last_ego_reference_wait_ok = False
        self.last_reach_candidates_wait_ok = False
        self.global_path_param_cache_used = False
        self.ego_reference_param_cache_used = False
        self.reach_candidates_param_cache_used = False
        self.last_failure_info = {}
        self.last_selected_candidate_detail = {}
        self.last_reference_repair_info = {}
        self._slot_constraints_cache = None
        self._point_margin_cache = {}
        self._arm_margin_cache = {}
        self._coupling_cache = {}
        self._edge_safety_cache = {}

        uav_pose_topic = rospy.get_param("~uav_pose_topic", "/iris_0/world_pose")
        self.subscribers = []
        self.subscribers.append(rospy.Subscriber(uav_pose_topic, PoseStamped, self._pose_cb, queue_size=1, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber("/arm/current_theta_phi", Float32MultiArray, self._arm_cb, queue_size=10, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber("/planner/global_path", Path, self._global_path_cb, queue_size=1, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber("/planner/global_path_array", Float32MultiArray, self._global_path_array_cb, queue_size=1, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber("/planner/ego_reference_trajectory", Path, self._reference_cb, queue_size=1, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber("/planner/ego_reference_trajectory_array", Float32MultiArray, self._reference_array_cb, queue_size=1, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber("/planner/selected_reach_goal", PoseStamped, self._selected_goal_cb, queue_size=1, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber("/planner/reach_candidates", Float32MultiArray, self._reach_candidates_cb, queue_size=10, tcp_nodelay=True))
        if self.use_static_cloud:
            self.subscribers.append(
                rospy.Subscriber(
                    self.static_cloud_topic,
                    PointCloud2,
                    self._static_cloud_cb,
                    queue_size=1,
                    buff_size=16 * 1024 * 1024,
                    tcp_nodelay=True,
                )
            )

        self.path_pub = rospy.Publisher("/planner/uav_waypoints", Path, queue_size=1, latch=True)
        self.arm_pub = rospy.Publisher("/planner/arm_waypoints", Float32MultiArray, queue_size=1, latch=True)
        self.plan_pub = rospy.Publisher("/planner/task_plan", String, queue_size=1, latch=True)
        self.safety_pub = rospy.Publisher("/planner/safety_report", String, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/planner/task_plan_markers", MarkerArray, queue_size=1, latch=True)

        rospy.loginfo("whole_body_planner_node: scenario=%s waiting for integrated inputs", self.scenario_name)

    def _publish_runtime_status(self, status, **fields):
        pub = getattr(self, "plan_pub", None)
        if pub is None:
            return
        lines = [
            "status=%s" % status,
            "planner_version=%s" % PLANNER_VERSION,
            "method=ours_full_min",
            "scenario=%s" % self.scenario_name,
            "input_refresh_attempts=%d" % self.input_refresh_attempts,
            "global_path_received=%s" % str(self.global_path is not None and bool(self.global_path.poses)),
            "global_path_points=%d" % (len(self.global_path.poses) if self.global_path is not None else 0),
            "ego_reference_received=%s" % str(self.reference_path is not None and bool(self.reference_path.poses)),
            "ego_reference_points=%d" % (len(self.reference_path.poses) if self.reference_path is not None else 0),
            "reach_candidate_count=%d" % len(self.reach_candidates),
            "static_cloud_loaded=%s" % str(self.static_cloud_loaded),
            "static_cloud_esdf=%s" % getattr(self.static_occupancy, "distance_field_status", "unknown"),
        ]
        for key in sorted(fields):
            lines.append("%s=%s" % (key, fields[key]))
        pub.publish(String(data="\n".join(lines)))

    def _pose_cb(self, msg):
        self.current_pose = msg.pose

    def _arm_cb(self, msg):
        if len(msg.data) >= 2:
            self.current_theta_phi = [float(msg.data[0]), float(msg.data[1])]

    def _global_path_cb(self, msg):
        self.global_path = msg
        self.global_path_stamp = rospy.Time.now()
        self.global_path_cb_count += 1

    def _global_path_array_cb(self, msg):
        path = self._path_from_array_msg(msg)
        if path is None:
            return
        self.global_path = path
        self.global_path_stamp = rospy.Time.now()
        self.global_path_array_cb_count += 1

    def _reference_cb(self, msg):
        self.reference_path = msg
        self.reference_cb_count += 1

    def _reference_array_cb(self, msg):
        path = self._path_from_array_msg(msg)
        if path is None:
            return
        self.reference_path = path
        self.reference_array_cb_count += 1

    def _path_from_array_msg(self, msg):
        if len(msg.data) < 4:
            return None
        stride = 4
        if msg.layout.dim and len(msg.layout.dim) >= 2:
            stride = int(msg.layout.dim[1].size)
        if stride < 4:
            return None
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = rospy.Time.now()
        for offset in range(0, len(msg.data) - stride + 1, stride):
            pose = self._make_pose(float(msg.data[offset]), float(msg.data[offset + 1]), float(msg.data[offset + 2]), float(msg.data[offset + 3]))
            ps = PoseStamped()
            ps.header = path.header
            ps.pose = pose
            path.poses.append(ps)
        return path if path.poses else None

    def _reference_from_param_cache(self):
        try:
            cache = rospy.get_param("/planner/latest_ego_reference_array_cache", None)
        except Exception:
            return None
        if not isinstance(cache, dict):
            return None
        data = cache.get("data", [])
        if not isinstance(data, list) or len(data) < 8:
            return None
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="waypoint", size=int(cache.get("count", len(data) // 4)), stride=len(data)),
            MultiArrayDimension(label="xyzyaw", size=4, stride=4),
        ]
        msg.layout.data_offset = 0
        msg.data = [float(value) for value in data]
        return self._path_from_array_msg(msg)

    def _global_path_from_param_cache(self):
        try:
            cache = rospy.get_param("/planner/latest_global_path_array_cache", None)
        except Exception:
            return None
        if not isinstance(cache, dict):
            return None
        data = cache.get("data", [])
        if not isinstance(data, list) or len(data) < 8:
            return None
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="waypoint", size=int(cache.get("count", len(data) // 4)), stride=len(data)),
            MultiArrayDimension(label="xyzyaw", size=4, stride=4),
        ]
        msg.layout.data_offset = 0
        msg.data = [float(value) for value in data]
        return self._path_from_array_msg(msg)

    def _reach_candidates_from_param_cache(self):
        try:
            cache = rospy.get_param("/planner/latest_reach_candidates_cache", None)
        except Exception:
            return []
        if not isinstance(cache, dict):
            return []
        data = cache.get("data", [])
        stride = int(cache.get("stride", 9))
        if not isinstance(data, list) or len(data) < stride or stride < 9:
            return []
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="candidate", size=int(cache.get("count", len(data) // stride)), stride=len(data)),
            MultiArrayDimension(label="x_y_z_yaw_theta_phi_tip_error_body_dist_score", size=stride, stride=stride),
        ]
        msg.layout.data_offset = 0
        msg.data = [float(value) for value in data]
        self._reach_candidates_cb(msg)
        return self.reach_candidates

    def _selected_goal_cb(self, msg):
        self.selected_goal = msg.pose

    def _static_cloud_cb(self, msg):
        if self.static_cloud_loaded:
            return
        raw_estimate = int(getattr(msg, "width", 0)) * int(getattr(msg, "height", 0))
        self._publish_runtime_status(
            "STATIC_CLOUD_UPDATE_BEGIN",
            cloud_frame=msg.header.frame_id,
            raw_point_estimate=raw_estimate,
        )
        started = time.time()
        points = []
        read_started = started
        for idx, point in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if idx % self.static_cloud_stride == 0:
                points.append((point[0], point[1], point[2]))
        read_elapsed = time.time() - read_started
        update_started = time.time()
        self.static_occupancy.update(points, msg.header.frame_id, msg.header.stamp)
        update_elapsed = time.time() - update_started
        self._clear_planning_caches()
        self.static_cloud_loaded = True
        total_elapsed = time.time() - started
        rospy.loginfo(
            "whole_body_planner_node: static cloud received points=%d occupied_voxels=%d frame=%s read=%.3fs update=%.3fs total=%.3fs %s",
            len(points),
            len(self.static_occupancy.occupied),
            msg.header.frame_id,
            read_elapsed,
            update_elapsed,
            total_elapsed,
            self.static_occupancy.report(),
        )
        self._publish_runtime_status(
            "STATIC_CLOUD_UPDATE_DONE",
            cloud_frame=msg.header.frame_id,
            raw_point_estimate=raw_estimate,
            sampled_points=len(points),
            occupied_voxels=len(self.static_occupancy.occupied),
            read_sec="%.3f" % read_elapsed,
            update_sec="%.3f" % update_elapsed,
            total_sec="%.3f" % total_elapsed,
        )

    def _clear_planning_caches(self):
        self._point_margin_cache = {}
        self._arm_margin_cache = {}
        self._coupling_cache = {}
        self._edge_safety_cache = {}

    def _reach_candidates_cb(self, msg):
        candidates = []
        stride = 9
        if msg.layout.dim and len(msg.layout.dim) >= 2:
            stride = int(msg.layout.dim[1].size)
        if stride < 9:
            return
        for offset in range(0, len(msg.data) - stride + 1, stride):
            values = [float(v) for v in msg.data[offset : offset + stride]]
            pose = self._make_pose(values[0], values[1], values[2], values[3])
            candidates.append(
                {
                    "pose": pose,
                    "theta_phi": [values[4], values[5]],
                    "tip_error": values[6],
                    "body_dist": values[7],
                    "score": values[8],
                }
            )
        candidates.sort(key=lambda c: c["score"])
        self.reach_candidates = candidates

    def _make_pose(self, x, y, z, yaw):
        pose = Pose()
        pose.position = point_msg(x, y, z)
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def _pose_xyz(self, pose):
        return (pose.position.x, pose.position.y, pose.position.z)

    def _pose_yaw(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return tft.euler_from_quaternion(q)[2]

    def _interp_yaw(self, a, b, ratio):
        delta = math.atan2(math.sin(b - a), math.cos(b - a))
        return a + ratio * delta

    def _copy_pose(self, pose):
        return self._make_pose(pose.position.x, pose.position.y, pose.position.z, self._pose_yaw(pose))

    def _normalize_angle_deg(self, value):
        return math.degrees(math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value))))

    def _interpolate_pose_between(self, a, b, ratio):
        ax, ay, az = self._pose_xyz(a)
        bx, by, bz = self._pose_xyz(b)
        ayaw = self._pose_yaw(a)
        byaw = self._pose_yaw(b)
        dyaw = math.atan2(math.sin(byaw - ayaw), math.cos(byaw - ayaw))
        return self._make_pose(
            ax + (bx - ax) * ratio,
            ay + (by - ay) * ratio,
            az + (bz - az) * ratio,
            ayaw + dyaw * ratio,
        )

    def _safe_pose_for_interpolated_shape(self, prev_pose, next_pose, theta, phi):
        for ratio in (0.0, 1.0, 0.5, 0.25, 0.75):
            pose = self._copy_pose(prev_pose) if ratio <= 1e-9 else self._copy_pose(next_pose) if ratio >= 1.0 - 1e-9 else self._interpolate_pose_between(prev_pose, next_pose, ratio)
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            arm_margin, _, _ = self._arm_margin_detail(pose, theta, phi)
            if body_margin >= self.required_body_margin and arm_margin >= self.required_arm_margin:
                return pose
        return None

    def _monotonic_arm_interpolation(self, prev_pose, next_pose, prev_shape, next_shape, steps):
        prev_theta, prev_phi = prev_shape
        next_theta, next_phi = next_shape
        dtheta = next_theta - prev_theta
        dphi = self._angle_delta(prev_phi, next_phi)
        poses = []
        shapes = []
        last_ratio = 0.0
        for step in range(1, steps):
            preferred_ratio = float(step) / float(steps)
            theta = prev_theta + dtheta * preferred_ratio
            phi = self._normalize_angle_deg(prev_phi + dphi * preferred_ratio)
            ratios = sorted(
                set(max(last_ratio, min(1.0, ratio)) for ratio in (preferred_ratio, last_ratio, 0.25, 0.5, 0.75, 1.0)),
                key=lambda ratio: (abs(ratio - preferred_ratio), ratio),
            )
            selected = None
            for ratio in ratios:
                pose = self._copy_pose(next_pose) if ratio >= 1.0 - 1e-9 else self._interpolate_pose_between(prev_pose, next_pose, ratio)
                body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
                arm_margin, _, _ = self._arm_margin_detail(pose, theta, phi)
                if body_margin >= self.required_body_margin and arm_margin >= self.required_arm_margin:
                    selected = (pose, ratio)
                    break
            if selected is None:
                return None
            poses.append(selected[0])
            shapes.append((theta, phi))
            last_ratio = selected[1]

        edge_poses = [prev_pose] + poses + [next_pose]
        edge_shapes = [prev_shape] + shapes + [next_shape]
        for edge_idx in range(1, len(edge_poses)):
            body_margin, arm_margin = self._seed_edge_margin(
                edge_poses[edge_idx - 1],
                edge_poses[edge_idx],
                edge_shapes[edge_idx - 1],
                edge_shapes[edge_idx],
            )
            if body_margin < self.required_body_margin or arm_margin < self.required_arm_margin:
                return None
        return poses, shapes

    def _interpolate_arm_limited(self, poses, arm_shapes):
        info = {
            "arm_interpolation_inserted": 0,
            "arm_interpolation_skipped_unsafe": 0,
            "arm_interpolation_monotonic_edges": 0,
            "arm_interpolation_fallback_edges": 0,
            "theta_limit": self.max_delta_theta,
            "phi_limit": self.max_delta_phi,
        }
        if not poses or not arm_shapes or len(poses) != len(arm_shapes):
            return poses, arm_shapes, info
        theta_limit = max(self.max_delta_theta, 1e-3)
        phi_limit = max(self.max_delta_phi, 1e-3)
        out_poses = [self._copy_pose(poses[0])]
        out_shapes = [arm_shapes[0]]
        for idx in range(1, len(poses)):
            prev_theta, prev_phi = out_shapes[-1]
            next_theta, next_phi = arm_shapes[idx]
            dtheta = next_theta - prev_theta
            dphi = self._angle_delta(prev_phi, next_phi)
            steps = max(1, int(math.ceil(abs(dtheta) / theta_limit)), int(math.ceil(abs(dphi) / phi_limit)))
            monotonic = self._monotonic_arm_interpolation(
                poses[idx - 1],
                poses[idx],
                (prev_theta, prev_phi),
                (next_theta, next_phi),
                steps,
            )
            if monotonic is not None:
                monotonic_poses, monotonic_shapes = monotonic
                out_poses.extend(monotonic_poses)
                out_shapes.extend(monotonic_shapes)
                info["arm_interpolation_inserted"] += len(monotonic_poses)
                info["arm_interpolation_monotonic_edges"] += 1
                out_poses.append(self._copy_pose(poses[idx]))
                out_shapes.append((next_theta, self._normalize_angle_deg(next_phi)))
                continue
            info["arm_interpolation_fallback_edges"] += 1
            for step in range(1, steps):
                ratio = float(step) / float(steps)
                theta = prev_theta + dtheta * ratio
                phi = self._normalize_angle_deg(prev_phi + dphi * ratio)
                pose = self._safe_pose_for_interpolated_shape(poses[idx - 1], poses[idx], theta, phi)
                if pose is None:
                    info["arm_interpolation_skipped_unsafe"] += 1
                    continue
                out_poses.append(pose)
                out_shapes.append((theta, phi))
                info["arm_interpolation_inserted"] += 1
            out_poses.append(self._copy_pose(poses[idx]))
            out_shapes.append((next_theta, self._normalize_angle_deg(next_phi)))
        return out_poses, out_shapes, info

    def _visual_arm_samples(self, poses, arm_shapes):
        if not poses or not arm_shapes or len(poses) != len(arm_shapes):
            return [], [], []
        out_poses = [self._copy_pose(poses[0])]
        out_shapes = [arm_shapes[0]]
        out_progress = [0.0]
        for idx in range(1, len(poses)):
            prev_pose = poses[idx - 1]
            next_pose = poses[idx]
            prev_theta, prev_phi = arm_shapes[idx - 1]
            next_theta, next_phi = arm_shapes[idx]
            dist = point_distance(self._pose_xyz(prev_pose), self._pose_xyz(next_pose))
            dtheta = next_theta - prev_theta
            dphi = self._angle_delta(prev_phi, next_phi)
            steps = max(
                1,
                int(math.ceil(dist / self.marker_arm_visual_spacing)),
                int(math.ceil(abs(dtheta) / self.marker_arm_visual_theta_step)),
                int(math.ceil(abs(dphi) / self.marker_arm_visual_phi_step)),
            )
            for step in range(1, steps + 1):
                ratio = float(step) / float(steps)
                out_poses.append(self._interpolate_pose_between(prev_pose, next_pose, ratio))
                out_shapes.append(
                    (
                        prev_theta + dtheta * ratio,
                        self._normalize_angle_deg(prev_phi + dphi * ratio),
                    )
                )
                out_progress.append(float(idx - 1) + ratio)
        return out_poses, out_shapes, out_progress

    def _box_margin(self, point, xy_clearance, z_clearance=None):
        best = float("inf")
        name = "none"
        if z_clearance is None:
            z_clearance = xy_clearance
        for box in self.boxes:
            margin = self._expanded_box_margin(point, box, xy_clearance, z_clearance)
            if margin < best:
                best = margin
                name = box.name
        return best, name

    def _expanded_box_margin(self, point, box, xy_clearance, z_clearance):
        dx = abs(point[0] - box.center[0]) - (box.half[0] + xy_clearance)
        dy = abs(point[1] - box.center[1]) - (box.half[1] + xy_clearance)
        dz = abs(point[2] - box.center[2]) - (box.half[2] + z_clearance)
        ox = max(dx, 0.0)
        oy = max(dy, 0.0)
        oz = max(dz, 0.0)
        outside = math.sqrt(ox * ox + oy * oy + oz * oz)
        inside = min(max(dx, max(dy, dz)), 0.0)
        return outside + inside

    def _keepout_margin(self, point):
        best = float("inf")
        name = "none"
        for box in self.body_keepout_boxes:
            margin = box.signed_distance(point)
            if margin < best:
                best = margin
                name = box.name
        return best, name

    def _cloud_margin(self, point, clearance):
        if not self.use_static_cloud or not self.cloud_collision_enabled:
            return float("inf"), "static_cloud:disabled", point
        if self.require_static_cloud and not self.static_occupancy.ready():
            return -float("inf"), "static_cloud:not_ready", point
        if not self.static_occupancy.ready():
            return float("inf"), "static_cloud:not_ready", point
        # For collision checking we only need nearby obstacles. Searching the
        # full visualization/global radius inside every whole-body candidate is
        # the main source of multi-minute stalls.
        query_radius = min(
            max(clearance + self.static_cloud_query_padding, self.static_occupancy.resolution),
            self.static_cloud_search_radius,
        )
        dist, _, nearest = self.static_occupancy.signed_distance(point, query_radius)
        return dist - clearance, "static_scene_cloud", nearest

    def _point_margin(self, point, clearance, include_keepout=False, z_clearance=None):
        cache_key = (
            round(float(point[0]), 3),
            round(float(point[1]), 3),
            round(float(point[2]), 3),
            round(float(clearance), 3),
            bool(include_keepout),
            round(float(z_clearance if z_clearance is not None else clearance), 3),
            bool(self.cloud_collision_enabled),
        )
        cached = self._point_margin_cache.get(cache_key)
        if cached is not None:
            return cached
        box_margin, box_name = self._box_margin(point, clearance, z_clearance)
        cloud_margin, cloud_name, cloud_point = self._cloud_margin(point, clearance)
        best_margin, best_name, best_point = box_margin, box_name, point
        if cloud_margin < best_margin:
            best_margin, best_name, best_point = cloud_margin, cloud_name, cloud_point
        if include_keepout:
            keepout_margin, keepout_name = self._keepout_margin(point)
            if keepout_margin < best_margin:
                best_margin, best_name, best_point = keepout_margin, keepout_name, point
        result = (best_margin, best_name, best_point)
        if len(self._point_margin_cache) < 500000:
            self._point_margin_cache[cache_key] = result
        return result

    def _body_point_margin(self, point):
        return self._point_margin(
            point,
            self.body_radius + self.body_clearance,
            True,
            0.5 * self.body_height + self.body_clearance,
        )

    def _body_safe(self, pose):
        margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
        return margin >= 0.0

    def _arm_safe(self, pose, theta, phi):
        threshold = self.arm_params["radius"] + self.arm_clearance
        for point in world_arm_points(pose, theta, phi, self.arm_params):
            margin, _, _ = self._point_margin(point, threshold, False)
            if margin < 0.0:
                return False
        return True

    def _reach_in_swept_margin(self, pose, start_shape, goal_shape):
        dtheta = abs(goal_shape[0] - start_shape[0])
        dphi = abs(self._angle_delta(start_shape[1], goal_shape[1]))
        steps = max(
            2,
            int(math.ceil(dtheta / max(self.swept_theta_step, 1e-3))),
            int(math.ceil(dphi / max(self.swept_phi_step, 1e-3))),
        )
        best = float("inf")
        best_obstacle = "none"
        best_point = (0.0, 0.0, 0.0)
        for step in range(steps + 1):
            ratio = float(step) / float(steps)
            theta = start_shape[0] + (goal_shape[0] - start_shape[0]) * ratio
            phi = self._normalize_angle_deg(start_shape[1] + self._angle_delta(start_shape[1], goal_shape[1]) * ratio)
            margin, obstacle, point = self._arm_margin_detail(pose, theta, phi)
            if margin < best:
                best = margin
                best_obstacle = obstacle
                best_point = point
        return best, best_obstacle, best_point

    def _reach_candidate_score(self, candidate, selected_xyz=None):
        pose = candidate["pose"]
        goal_shape = (candidate["theta_phi"][0], candidate["theta_phi"][1])
        body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
        goal_arm_margin, goal_arm_obstacle, goal_arm_point = self._arm_margin_detail(pose, goal_shape[0], goal_shape[1])
        swept_arm_margin, swept_arm_obstacle, swept_arm_point = self._reach_in_swept_margin(
            pose,
            (self.compact_theta, self.compact_phi),
            goal_shape,
        )
        body_target = max(self.standoff_body_margin_target, self.refine_clearance_target)
        arm_target = max(self.standoff_arm_margin_target, self.refine_clearance_target)
        body_deficit = max(0.0, body_target - body_margin) / max(body_target, 1e-3)
        arm_deficit = max(0.0, arm_target - min(goal_arm_margin, swept_arm_margin)) / max(arm_target, 1e-3)
        selected_dist = 0.0
        if selected_xyz is not None:
            selected_dist = point_distance(selected_xyz, self._pose_xyz(pose))
        hard_penalty = 0.0
        if body_margin < self.required_body_margin:
            hard_penalty += 1000.0
        if goal_arm_margin < self.required_arm_margin or swept_arm_margin < self.required_arm_margin:
            hard_penalty += 1000.0
        if body_margin < body_target:
            hard_penalty += 250.0 * ((body_target - body_margin) / max(body_target, 1e-3)) ** 2
        if min(goal_arm_margin, swept_arm_margin) < arm_target:
            hard_penalty += 250.0 * ((arm_target - min(goal_arm_margin, swept_arm_margin)) / max(arm_target, 1e-3)) ** 2
        score = (
            float(candidate.get("score", 0.0))
            + 0.8 * selected_dist
            + 80.0 * body_deficit * body_deficit
            + 120.0 * arm_deficit * arm_deficit
            + hard_penalty
        )
        detail = {
            "body_margin": body_margin,
            "body_obstacle": body_obstacle,
            "body_point": body_point,
            "goal_arm_margin": goal_arm_margin,
            "goal_arm_obstacle": goal_arm_obstacle,
            "goal_arm_point": goal_arm_point,
            "swept_arm_margin": swept_arm_margin,
            "swept_arm_obstacle": swept_arm_obstacle,
            "swept_arm_point": swept_arm_point,
            "selected_dist": selected_dist,
            "score": score,
            "standoff_xyz": self._pose_xyz(pose),
        }
        return score, detail

    def _standoff_approach_score(self, start_pose, standoff_pose):
        lx, ly, lz = self._pose_xyz(start_pose)
        sx, sy, sz = self._pose_xyz(standoff_pose)
        dist = point_distance((lx, ly, lz), (sx, sy, sz))
        if dist < 0.25:
            return {
                "score": 0.0,
                "body_margin": float("inf"),
                "arm_margin": float("inf"),
                "hard_unsafe": False,
            }
        dx = lx - sx
        dy = ly - sy
        dz = lz - sz
        norm = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
        ux = dx / norm
        uy = dy / norm
        px = -uy
        py = ux
        max_step = min(self.standoff_approach_distance, max(0.35, 0.85 * dist))
        step_values = sorted(
            set(
                [
                    min(max_step, 0.35),
                    min(max_step, 0.50),
                    min(max_step, 0.65),
                    max_step,
                ]
            )
        )
        lateral_values = [0.0, -0.20, 0.20, -0.40, 0.40]
        z_values = [
            sz + self.standoff_approach_z_lift,
            sz + self.standoff_approach_z_lift + 0.12,
            sz + self.standoff_approach_z_lift - 0.10,
            sz + 0.04,
            max(sz + 0.04, min(max(lz, sz), sz + 0.34)),
        ]
        best = {
            "score": float("inf"),
            "body_margin": -float("inf"),
            "arm_margin": -float("inf"),
            "hard_unsafe": True,
        }
        for step in step_values:
            for lateral in lateral_values:
                for z in z_values:
                    approach_x = sx + ux * step + px * lateral
                    approach_y = sy + uy * step + py * lateral
                    approach_z = max(0.90, min(2.05, z))
                    approach_yaw = math.atan2(sy - approach_y, sx - approach_x)
                    approach_pose = self._make_pose(approach_x, approach_y, approach_z, approach_yaw)
                    point_body, _, _ = self._body_point_margin(self._pose_xyz(approach_pose))
                    point_arm = self._arm_margin(approach_pose, self.compact_theta, self.compact_phi, sample_limit=7)
                    edge0_body, edge0_arm = self._seed_edge_margin(start_pose, approach_pose)
                    edge1_body, edge1_arm = self._seed_edge_margin(approach_pose, standoff_pose)
                    min_body = min(point_body, edge0_body, edge1_body)
                    min_arm = min(point_arm, edge0_arm, edge1_arm)
                    body_deficit = max(0.0, self.sequence_risk_body_margin - min_body) / max(self.sequence_risk_body_margin, 1e-3)
                    arm_deficit = max(0.0, self.sequence_risk_arm_margin - min_arm) / max(self.sequence_risk_arm_margin, 1e-3)
                    hard_unsafe = min_body < self.required_body_margin or min_arm < self.required_arm_margin
                    score = (
                        800.0 * float(hard_unsafe)
                        + 90.0 * body_deficit * body_deficit
                        + 90.0 * arm_deficit * arm_deficit
                        + 0.25 * abs(lateral)
                        + 0.12 * abs(approach_z - sz)
                        - 4.0 * min(min_body, self.sequence_risk_body_margin)
                        - 4.0 * min(min_arm, self.sequence_risk_arm_margin)
                    )
                    if score < best["score"]:
                        best = {
                            "score": score,
                            "body_margin": min_body,
                            "arm_margin": min_arm,
                            "hard_unsafe": hard_unsafe,
                        }
        return best

    def _approach_start_for_standoff(self, reference_poses, standoff_pose):
        if not reference_poses:
            return None
        sx, sy, sz = self._pose_xyz(standoff_pose)
        min_dist = max(0.80, self.standoff_approach_distance)
        for pose in reversed(reference_poses):
            if point_distance(self._pose_xyz(pose), (sx, sy, sz)) >= min_dist:
                return pose
        return reference_poses[0]

    def _find_candidate_for_selected_goal(self, approach_reference_poses=None):
        if not self.reach_candidates:
            return None, -1
        selected = self._pose_xyz(self.selected_goal) if self.selected_goal is not None else None
        scored = []
        candidates_to_score = self.reach_candidates[: min(len(self.reach_candidates), self.standoff_candidate_score_count)]
        self._publish_runtime_status(
            "SELECT_REACH_CANDIDATE_SCORE_BEGIN",
            reach_candidates_total=len(self.reach_candidates),
            reach_candidates_scored=len(candidates_to_score),
            approach_rerank_limit=self.standoff_approach_rerank_count,
        )
        score_started = time.time()
        selection_deadline = score_started + self.standoff_selection_time_budget if self.standoff_selection_time_budget > 0.0 else float("inf")
        selection_timed_out = False
        for local_idx, candidate in enumerate(candidates_to_score):
            if time.time() >= selection_deadline and scored:
                selection_timed_out = True
                break
            idx = local_idx
            if local_idx > 0 and local_idx % 25 == 0:
                self._publish_runtime_status(
                    "SELECT_REACH_CANDIDATE_SCORE_PROGRESS",
                    scored_candidates=local_idx,
                    reach_candidates_scored=len(candidates_to_score),
                    elapsed_sec="%.3f" % (time.time() - score_started),
                )
            score, detail = self._reach_candidate_score(candidate, selected)
            strict = (
                detail["body_margin"] >= self.standoff_body_margin_target
                and detail["goal_arm_margin"] >= self.standoff_arm_margin_target
                and detail["swept_arm_margin"] >= self.standoff_arm_margin_target
            )
            hard_safe = (
                detail["body_margin"] >= self.required_body_margin
                and detail["goal_arm_margin"] >= self.required_arm_margin
                and detail["swept_arm_margin"] >= self.required_arm_margin
            )
            detail["standoff_strict_safe"] = strict
            detail["standoff_hard_safe"] = hard_safe
            if strict:
                safety_rank = 0
            elif hard_safe:
                safety_rank = 1
            else:
                safety_rank = 2
            scored.append((safety_rank, score, idx, candidate, detail))
        scored.sort(key=lambda item: (item[0], item[1]))
        strict_count = sum(1 for item in scored if item[0] == 0)
        hard_safe_count = sum(1 for item in scored if item[0] <= 1)
        self._publish_runtime_status(
            "SELECT_REACH_CANDIDATE_SCORE_DONE",
            scored_candidates=len(scored),
            strict_candidate_count=strict_count,
            selection_timed_out=str(selection_timed_out),
            elapsed_sec="%.3f" % (time.time() - score_started),
        )
        if approach_reference_poses is not None and scored and time.time() < selection_deadline:
            rerank_count = min(len(scored), self.standoff_approach_rerank_count)
            reranked = []
            rerank_started = time.time()
            self._publish_runtime_status(
                "SELECT_REACH_CANDIDATE_RERANK_BEGIN",
                rerank_count=rerank_count,
                scored_candidates=len(scored),
                remaining_budget_sec="%.3f" % max(0.0, selection_deadline - time.time()),
            )
            for strict_rank, score, idx, candidate, detail in scored[:rerank_count]:
                if time.time() >= selection_deadline and reranked:
                    selection_timed_out = True
                    break
                approach_start_pose = self._approach_start_for_standoff(approach_reference_poses, candidate["pose"])
                if approach_start_pose is None:
                    continue
                approach = self._standoff_approach_score(approach_start_pose, candidate["pose"])
                detail["approach_score"] = approach["score"]
                detail["approach_body_margin"] = approach["body_margin"]
                detail["approach_arm_margin"] = approach["arm_margin"]
                detail["approach_hard_unsafe"] = approach["hard_unsafe"]
                approach_rank = 1 if approach["hard_unsafe"] else 0
                combined = score + approach["score"]
                reranked.append((strict_rank, approach_rank, combined, idx, candidate, detail))
            self._publish_runtime_status(
                "SELECT_REACH_CANDIDATE_RERANK_DONE",
                reranked_candidates=len(reranked),
                selection_timed_out=str(selection_timed_out),
                elapsed_sec="%.3f" % (time.time() - rerank_started),
            )
            selectable_reranked = [item for item in reranked if item[0] <= 1] if hard_safe_count > 0 else reranked
            safe_reranked = [item for item in selectable_reranked if item[1] == 0]
            selected_pool = safe_reranked if safe_reranked else selectable_reranked
            selected_pool.sort(key=lambda item: (item[0], item[2]))
            if selected_pool:
                strict_rank, _, combined, best_idx, best_candidate, best_detail = selected_pool[0]
                best_detail["score"] = combined
                best_detail["strict_candidate_count"] = strict_count
                best_detail["hard_safe_candidate_count"] = hard_safe_count
                best_detail["approach_safe_candidate_count"] = len(safe_reranked)
                best_detail["scored_candidate_count"] = len(scored)
                best_detail["reranked_candidate_count"] = len(reranked)
                best_detail["selection_timed_out"] = selection_timed_out
                selected_candidate = dict(best_candidate)
                selected_candidate["selection_detail"] = best_detail or {}
                return selected_candidate, best_idx
        elif approach_reference_poses is not None and scored:
            selection_timed_out = True
            self._publish_runtime_status(
                "SELECT_REACH_CANDIDATE_RERANK_SKIPPED",
                scored_candidates=len(scored),
                selection_timed_out=str(selection_timed_out),
                elapsed_sec="%.3f" % (time.time() - score_started),
            )
        if not scored:
            return None, -1
        _, _, best_idx, best_candidate, best_detail = scored[0]
        best_detail["strict_candidate_count"] = strict_count
        best_detail["hard_safe_candidate_count"] = hard_safe_count
        best_detail["scored_candidate_count"] = len(scored)
        best_detail["reranked_candidate_count"] = 0
        best_detail["selection_timed_out"] = selection_timed_out
        if best_idx < 0:
            return None, -1
        selected_candidate = dict(best_candidate)
        selected_candidate["selection_detail"] = best_detail or {}
        return selected_candidate, best_idx

    def _path_poses(self, path_msg):
        return [self._copy_pose(ps.pose) for ps in path_msg.poses]

    def _repair_reference_with_global_path(self, reference_poses):
        info = {
            "ego_reference_points": len(reference_poses),
            "unsafe_ego_points": 0,
            "global_repair_points": 0,
            "global_repair_used": False,
            "first_unsafe_ego_idx": -1,
            "first_unsafe_ego_margin": float("inf"),
            "first_unsafe_ego_obstacle": "none",
            "first_unsafe_ego_pose": (0.0, 0.0, 0.0),
            "first_unsafe_ego_point": (0.0, 0.0, 0.0),
        }
        if not reference_poses:
            return reference_poses, info
        repaired = []
        first_unsafe_idx = None
        for idx, pose in enumerate(reference_poses):
            margin, obstacle, point = self._body_point_margin(self._pose_xyz(pose))
            if margin >= 0.0:
                repaired.append(pose)
                continue
            first_unsafe_idx = idx
            info["first_unsafe_ego_idx"] = idx
            info["first_unsafe_ego_margin"] = margin
            info["first_unsafe_ego_obstacle"] = obstacle
            info["first_unsafe_ego_pose"] = self._pose_xyz(pose)
            info["first_unsafe_ego_point"] = point
            break
        if first_unsafe_idx is None:
            return repaired, info

        info["unsafe_ego_points"] = len(reference_poses) - first_unsafe_idx
        if self.global_path is None or not self.global_path.poses:
            return repaired, info

        anchor_pose = repaired[-1] if repaired else None
        if anchor_pose is None and self.current_pose is not None:
            anchor_pose = self._copy_pose(self.current_pose)
        if anchor_pose is None:
            return repaired, info

        global_poses = self._path_poses(self.global_path)
        anchor = self._pose_xyz(anchor_pose)
        best_idx = 0
        best_dist = float("inf")
        for idx, pose in enumerate(global_poses):
            dist = point_distance(anchor, self._pose_xyz(pose))
            if dist < best_dist:
                best_idx = idx
                best_dist = dist

        for pose in global_poses[best_idx:]:
            if not self._body_safe(pose):
                continue
            if repaired and point_distance(self._pose_xyz(repaired[-1]), self._pose_xyz(pose)) < 0.05:
                continue
            repaired.append(pose)
            info["global_repair_points"] += 1
        info["global_repair_used"] = info["global_repair_points"] > 0
        return repaired, info

    def _slot_constraints(self):
        if self._slot_constraints_cache is not None:
            return self._slot_constraints_cache
        slots = []
        body_z_clearance = 0.5 * self.body_height + self.body_clearance
        min_horizontal_overlap = max(2.0, 4.0 * self.body_radius)
        min_gap = 2.0 * body_z_clearance + 2.0 * self.required_body_margin
        max_gap = 1.25
        for lower in self.boxes:
            lower_top = lower.center[2] + lower.half[2]
            for ceiling in self.boxes:
                if lower is ceiling:
                    continue
                ceiling_bottom = ceiling.center[2] - ceiling.half[2]
                gap = ceiling_bottom - lower_top
                if gap < min_gap or gap > max_gap:
                    continue
                x_min = max(lower.center[0] - lower.half[0], ceiling.center[0] - ceiling.half[0])
                x_max = min(lower.center[0] + lower.half[0], ceiling.center[0] + ceiling.half[0])
                y_min = max(lower.center[1] - lower.half[1], ceiling.center[1] - ceiling.half[1])
                y_max = min(lower.center[1] + lower.half[1], ceiling.center[1] + ceiling.half[1])
                x_overlap = x_max - x_min
                y_overlap = y_max - y_min
                if x_overlap < min_horizontal_overlap or y_overlap <= 0.02:
                    continue
                z_low = lower_top + body_z_clearance + self.required_body_margin
                z_high = ceiling_bottom - body_z_clearance - self.required_body_margin
                if z_high < z_low:
                    continue
                slot_id = "slot_%d" % len(slots)
                z_target = min(z_high, max(z_low, z_high - 0.02))
                slots.append(
                    {
                        "id": slot_id,
                        "lower_name": lower.name,
                        "ceiling_name": ceiling.name,
                        "y": 0.5 * (y_min + y_max),
                        "half_y": 0.5 * y_overlap,
                        "x_min": x_min,
                        "x_max": x_max,
                        "z_low": z_low,
                        "z_high": z_high,
                        "z_target": z_target,
                    }
                )
        self._slot_constraints_cache = slots
        return slots

    def _slot_constraint_for_pose(self, pose):
        x, y, _ = self._pose_xyz(pose)
        best = None
        best_dist = float("inf")
        for slot in self._slot_constraints():
            if x < slot["x_min"] - 0.60 or x > slot["x_max"] + 0.60:
                continue
            dist = abs(y - slot["y"])
            band = slot["half_y"] + max(0.35, 2.0 * self.refine_lateral_step)
            if dist <= band and dist < best_dist:
                best = slot
                best_dist = dist
        return best

    def _project_pose_to_slot_window(self, pose):
        slot = self._slot_constraint_for_pose(pose)
        if slot is None:
            return self._copy_pose(pose), None, False
        x, y, z = self._pose_xyz(pose)
        target_z = slot["z_target"]
        changed = abs(z - target_z) > 0.015
        if not changed:
            return self._copy_pose(pose), slot, False
        return self._make_pose(x, y, target_z, self._pose_yaw(pose)), slot, True

    def _repair_reference_with_slot_keyframes(self, poses):
        info = {
            "slot_keyframes_inserted": 0,
            "slot_poses_projected": 0,
            "slot_keyframe_constraints": [],
        }
        if len(poses) < 2:
            return list(poses), info
        slots = self._slot_constraints()
        if not slots:
            return list(poses), info
        out = []
        inserted_keys = set()
        constrained_slots = set()
        for idx in range(len(poses) - 1):
            a = poses[idx]
            b = poses[idx + 1]
            projected_a, _, projected = self._project_pose_to_slot_window(a)
            if projected:
                info["slot_poses_projected"] += 1
            out.append(projected_a)
            ax, ay, _ = self._pose_xyz(a)
            bx, by, _ = self._pose_xyz(b)
            dy = by - ay
            if abs(dy) < 1e-6:
                continue
            crossings = []
            for slot in slots:
                guard = slot["half_y"] + max(0.12, self.body_radius + self.body_clearance)
                for target_idx, target_y in enumerate((slot["y"] - guard, slot["y"], slot["y"] + guard)):
                    if (ay - target_y) * (by - target_y) > 0.0:
                        continue
                    ratio = (target_y - ay) / dy
                    if ratio <= 1e-4 or ratio >= 1.0 - 1e-4:
                        continue
                    x_at_target = ax + (bx - ax) * ratio
                    if x_at_target < slot["x_min"] - 0.60 or x_at_target > slot["x_max"] + 0.60:
                        continue
                    crossings.append((ratio, slot, target_idx, target_y, x_at_target))
            crossings.sort(key=lambda item: item[0])
            for _, slot, target_idx, target_y, x_value in crossings:
                key = (slot["id"], target_idx)
                if key in inserted_keys:
                    continue
                inserted_keys.add(key)
                yaw = math.atan2(by - ay, bx - ax)
                pose = self._make_pose(x_value, target_y, slot["z_target"], yaw)
                if out and point_distance(self._pose_xyz(out[-1]), self._pose_xyz(pose)) < 0.05:
                    continue
                out.append(pose)
                info["slot_keyframes_inserted"] += 1
                if slot["id"] in constrained_slots:
                    continue
                constrained_slots.add(slot["id"])
                info["slot_keyframe_constraints"].append(
                    "%s:%s/%s:z=%.3f[%.3f,%.3f]"
                    % (slot["id"], slot["lower_name"], slot["ceiling_name"], slot["z_target"], slot["z_low"], slot["z_high"])
                )
        projected_last, _, projected = self._project_pose_to_slot_window(poses[-1])
        if projected:
            info["slot_poses_projected"] += 1
        out.append(projected_last)
        return out, info

    def _append_standoff_if_needed(self, poses, standoff_pose):
        if not poses:
            return [self._copy_pose(standoff_pose)]
        if point_distance(self._pose_xyz(poses[-1]), self._pose_xyz(standoff_pose)) <= 0.05:
            poses[-1] = self._copy_pose(standoff_pose)
            return poses
        poses.append(self._copy_pose(standoff_pose))
        return poses

    def _seed_edge_margin(self, a, b, shape_a=None, shape_b=None):
        if shape_a is None:
            shape_a = (self.compact_theta, self.compact_phi)
        if shape_b is None:
            shape_b = shape_a
        dist = point_distance(self._pose_xyz(a), self._pose_xyz(b))
        dtheta = abs(shape_b[0] - shape_a[0])
        dphi = abs(self._angle_delta(shape_a[1], shape_b[1]))
        steps = max(
            2,
            int(math.ceil(dist / self.swept_translation_step)),
            int(math.ceil(dtheta / self.swept_theta_step)),
            int(math.ceil(dphi / self.swept_phi_step)),
        )
        min_body = float("inf")
        min_arm = float("inf")
        for step in range(steps + 1):
            ratio = float(step) / float(steps)
            pose = self._interpolate_pose_between(a, b, ratio)
            theta = shape_a[0] + (shape_b[0] - shape_a[0]) * ratio
            phi = self._normalize_angle_deg(shape_a[1] + self._angle_delta(shape_a[1], shape_b[1]) * ratio)
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            arm_margin, _, _ = self._arm_margin_detail(pose, theta, phi, sample_limit=7)
            min_body = min(min_body, body_margin)
            min_arm = min(min_arm, arm_margin)
        return min_body, min_arm

    def _reference_edge_detour_pose(self, a, b):
        mid = self._interpolate_pose_between(a, b, 0.5)
        mid, _, _ = self._project_pose_to_slot_window(mid)
        ax, ay, az = self._pose_xyz(a)
        bx, by, bz = self._pose_xyz(b)
        yaw = math.atan2(by - ay, bx - ax)
        left = (-math.sin(yaw), math.cos(yaw))
        offsets = [
            (0.0, 0.0),
            (0.20, 0.0),
            (-0.20, 0.0),
            (0.35, 0.0),
            (-0.35, 0.0),
            (0.0, 0.16),
            (0.20, 0.12),
            (-0.20, 0.12),
            (0.35, 0.16),
            (-0.35, 0.16),
            (0.55, 0.12),
            (-0.55, 0.12),
            (0.75, 0.18),
            (-0.75, 0.18),
            (0.0, 0.24),
            (0.20, 0.24),
            (-0.20, 0.24),
            (0.0, -0.10),
        ]
        best = None
        best_score = -float("inf")
        mx, my, mz = self._pose_xyz(mid)
        for lateral, dz in offsets:
            pose = self._make_pose(mx + left[0] * lateral, my + left[1] * lateral, max(0.90, min(2.05, mz + dz)), yaw)
            pose, _, _ = self._project_pose_to_slot_window(pose)
            point_body, _, _ = self._body_point_margin(self._pose_xyz(pose))
            point_arm = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=7)
            edge1_body, edge1_arm = self._seed_edge_margin(a, pose)
            edge2_body, edge2_arm = self._seed_edge_margin(pose, b)
            body_margin = min(point_body, edge1_body, edge2_body)
            arm_margin = min(point_arm, edge1_arm, edge2_arm)
            # Skeleton connectivity is a UAV-body topology problem. The arm
            # shape is optimized later by whole-body DP, so compact-arm
            # clearance is only a weak preference here, not a hard gate.
            score = body_margin + 0.10 * min(arm_margin, self.refine_clearance_target) - 0.08 * abs(lateral) - 0.04 * abs(dz)
            if score > best_score:
                best = pose
                best_score = score
        if best is None:
            return None, -float("inf")
        return best, best_score

    def _transition_detour_pose(self, a, b):
        direct_body, direct_arm = self._seed_edge_margin(a, b)
        mid = self._interpolate_pose_between(a, b, 0.5)
        ax, ay, az = self._pose_xyz(a)
        bx, by, bz = self._pose_xyz(b)
        yaw = math.atan2(by - ay, bx - ax)
        left = (-math.sin(yaw), math.cos(yaw))
        mx, my, mz = self._pose_xyz(mid)
        offsets = []
        for lateral in (0.0, 0.18, -0.18, 0.35, -0.35, 0.55, -0.55, 0.75, -0.75):
            for dz in (0.0, 0.08, -0.08, 0.16, -0.16, 0.24, -0.24, 0.32):
                offsets.append((lateral, dz))
        best = None
        best_score = -float("inf")
        for lateral, dz in offsets:
            pose = self._make_pose(
                mx + left[0] * lateral,
                my + left[1] * lateral,
                max(0.90, min(2.05, mz + dz)),
                yaw,
            )
            pose, _, _ = self._project_pose_to_slot_window(pose)
            point_body, _, _ = self._body_point_margin(self._pose_xyz(pose))
            point_arm = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=7)
            edge1_body, edge1_arm = self._seed_edge_margin(a, pose)
            edge2_body, edge2_arm = self._seed_edge_margin(pose, b)
            body_margin = min(point_body, edge1_body, edge2_body)
            arm_margin = min(point_arm, edge1_arm, edge2_arm)
            score = (
                2.0 * min(body_margin, self.required_body_margin + 0.08)
                + min(arm_margin, self.required_arm_margin + 0.08)
                - 0.05 * abs(lateral)
                - 0.03 * abs(dz)
            )
            if score > best_score:
                best = pose
                best_score = score
        if best is None:
            return None
        best_body, best_arm = self._seed_edge_margin(a, best)
        best2_body, best2_arm = self._seed_edge_margin(best, b)
        if min(best_body, best2_body) <= direct_body and min(best_arm, best2_arm) <= direct_arm:
            return None
        return best

    def _slot_constraint_for_edge(self, a, b):
        candidates = []
        for pose in (a, self._interpolate_pose_between(a, b, 0.5), b):
            slot = self._slot_constraint_for_pose(pose)
            if slot is not None and all(item["id"] != slot["id"] for item in candidates):
                candidates.append(slot)
        if not candidates:
            ax, ay, _ = self._pose_xyz(a)
            bx, by, _ = self._pose_xyz(b)
            dy = by - ay
            if abs(dy) < 1e-6:
                return None
            for slot in self._slot_constraints():
                if (ay - slot["y"]) * (by - slot["y"]) > 0.0:
                    continue
                ratio = (slot["y"] - ay) / dy
                if ratio <= 0.0 or ratio >= 1.0:
                    continue
                x = ax + (bx - ax) * ratio
                if slot["x_min"] - 0.60 <= x <= slot["x_max"] + 0.60:
                    candidates.append(slot)
        if not candidates:
            return None
        mx, my, _ = self._pose_xyz(self._interpolate_pose_between(a, b, 0.5))
        return min(
            candidates,
            key=lambda slot: abs(my - slot["y"]) + 0.05 * max(0.0, slot["x_min"] - mx, mx - slot["x_max"]),
        )

    def _slot_projected_edge_ladder(self, a, b, min_knots=3, max_knots=5):
        slot = self._slot_constraint_for_edge(a, b)
        if slot is None:
            return []
        ax, ay, _ = self._pose_xyz(a)
        bx, by, _ = self._pose_xyz(b)
        yaw = math.atan2(by - ay, bx - ax)
        dist = point_distance(self._pose_xyz(a), self._pose_xyz(b))
        knot_count = int(math.ceil(dist / max(0.28, 1.5 * self.swept_translation_step))) - 1
        knot_count = max(int(min_knots), min(int(max_knots), knot_count))
        x_low = slot["x_min"] + self.body_radius + self.required_body_margin
        x_high = slot["x_max"] - self.body_radius - self.required_body_margin
        poses = []
        for idx in range(1, knot_count + 1):
            ratio = float(idx) / float(knot_count + 1)
            base = self._interpolate_pose_between(a, b, ratio)
            x, y, _ = self._pose_xyz(base)
            if x_low <= x_high:
                x = min(max(x, x_low), x_high)
            pose = self._make_pose(x, y, slot["z_target"], yaw)
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            if body_margin < self.required_body_margin:
                projected, _, _ = self._project_pose_to_slot_window(base)
                pose = projected
            if point_distance(self._pose_xyz(a), self._pose_xyz(pose)) < 0.04:
                continue
            if point_distance(self._pose_xyz(b), self._pose_xyz(pose)) < 0.04:
                continue
            if poses and point_distance(self._pose_xyz(poses[-1]), self._pose_xyz(pose)) < 0.04:
                continue
            poses.append(pose)
        return poses

    def _swept_detour_pose(self, a, b, shape_a, shape_b):
        direct_body, direct_arm = self._seed_edge_margin(a, b, shape_a, shape_b)
        mid = self._interpolate_pose_between(a, b, 0.5)
        ax, ay, _ = self._pose_xyz(a)
        bx, by, _ = self._pose_xyz(b)
        yaw = math.atan2(by - ay, bx - ax)
        left = (-math.sin(yaw), math.cos(yaw))
        mx, my, mz = self._pose_xyz(mid)
        mid_shape = (
            shape_a[0] + (shape_b[0] - shape_a[0]) * 0.5,
            self._normalize_angle_deg(shape_a[1] + self._angle_delta(shape_a[1], shape_b[1]) * 0.5),
        )
        offsets = []
        for lateral in (0.0, 0.16, -0.16, 0.32, -0.32, 0.50, -0.50, 0.70, -0.70):
            for dz in (0.0, 0.06, 0.12, 0.20, 0.30, -0.05):
                offsets.append((lateral, dz))
        best = None
        best_score = -float("inf")
        best_margin = min(direct_body, direct_arm)
        for lateral, dz in offsets:
            pose = self._make_pose(
                mx + left[0] * lateral,
                my + left[1] * lateral,
                max(0.90, min(2.05, mz + dz)),
                yaw,
            )
            pose, _, _ = self._project_pose_to_slot_window(pose)
            point_body, _, _ = self._body_point_margin(self._pose_xyz(pose))
            point_arm = self._arm_margin(pose, mid_shape[0], mid_shape[1], sample_limit=7)
            edge1_body, edge1_arm = self._seed_edge_margin(a, pose, shape_a, mid_shape)
            edge2_body, edge2_arm = self._seed_edge_margin(pose, b, mid_shape, shape_b)
            body_margin = min(point_body, edge1_body, edge2_body)
            arm_margin = min(point_arm, edge1_arm, edge2_arm)
            margin = min(body_margin, arm_margin)
            score = (
                3.0 * min(body_margin, self.required_body_margin + 0.08)
                + 2.0 * min(arm_margin, self.required_arm_margin + 0.08)
                - 0.04 * abs(lateral)
                - 0.025 * abs(dz)
            )
            if score > best_score:
                best = pose
                best_score = score
                best_margin = margin
        if best is None:
            return None
        if best_margin <= min(direct_body, direct_arm) + 1e-3:
            return None
        return best

    def _densify_reference_edges(self, poses):
        info = {
            "reference_edge_splits": 0,
            "reference_edge_detours": 0,
            "reference_edge_unresolved": 0,
            "reference_edge_min_body_margin": float("inf"),
            "reference_edge_min_arm_margin": float("inf"),
            "reference_edge_densify_timeout": False,
        }
        if len(poses) < 2:
            return list(poses), info
        out = [self._copy_pose(poses[0])]
        queue = [(self._copy_pose(poses[idx]), self._copy_pose(poses[idx + 1]), 0) for idx in range(len(poses) - 1)]
        max_splits = 24
        while queue:
            a, b, depth = queue.pop(0)
            edge_body, edge_arm = self._seed_edge_margin(a, b)
            info["reference_edge_min_body_margin"] = min(info["reference_edge_min_body_margin"], edge_body)
            info["reference_edge_min_arm_margin"] = min(info["reference_edge_min_arm_margin"], edge_arm)
            should_split = edge_body < self.required_body_margin or edge_arm < self.required_arm_margin
            if should_split and depth < 4 and info["reference_edge_splits"] < max_splits:
                mid, detour_score = self._reference_edge_detour_pose(a, b)
                if mid is not None and detour_score >= min(self.required_body_margin, self.required_arm_margin):
                    info["reference_edge_detours"] += 1
                    queue.insert(0, (mid, b, depth + 1))
                    queue.insert(0, (a, mid, depth + 1))
                    info["reference_edge_splits"] += 1
                    continue
                info["reference_edge_unresolved"] += 1
            if point_distance(self._pose_xyz(out[-1]), self._pose_xyz(b)) >= 0.03:
                out.append(self._copy_pose(b))
        return out, info

    def _measure_reference_edges(self, poses):
        info = {
            "reference_edge_splits": 0,
            "reference_edge_detours": 0,
            "reference_edge_unresolved": 0,
            "reference_edge_min_body_margin": float("inf"),
            "reference_edge_min_arm_margin": float("inf"),
        }
        if len(poses) < 2:
            return info
        for idx in range(len(poses) - 1):
            edge_body, edge_arm = self._seed_edge_margin(poses[idx], poses[idx + 1])
            info["reference_edge_min_body_margin"] = min(info["reference_edge_min_body_margin"], edge_body)
            info["reference_edge_min_arm_margin"] = min(info["reference_edge_min_arm_margin"], edge_arm)
            if edge_body < self.required_body_margin or edge_arm < self.required_arm_margin:
                info["reference_edge_unresolved"] += 1
        return info

    def _compress_reference_keyframes(self, poses):
        target = max(8, min(self.refine_max_points, self.sequence_refine_max_points, 24))
        info = {
            "reference_keyframes_pruned": 0,
            "reference_keyframes_after_compression": len(poses),
        }
        if len(poses) <= target:
            return list(poses), info

        def required_idx(items, idx):
            if idx <= 0 or idx >= len(items) - 1:
                return True
            if idx >= len(items) - 4:
                return True
            pose = items[idx]
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            if body_margin < max(self.refine_clearance_target, self.required_body_margin + 0.03):
                return True
            a = self._pose_xyz(items[idx - 1])
            b = self._pose_xyz(pose)
            c = self._pose_xyz(items[idx + 1])
            yaw1 = math.atan2(b[1] - a[1], b[0] - a[0])
            yaw2 = math.atan2(c[1] - b[1], c[0] - b[0])
            if abs(math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1))) > math.radians(25.0):
                return True
            if abs(items[idx + 1].position.z - items[idx - 1].position.z) > 0.22:
                return True
            return False

        out = [self._copy_pose(pose) for pose in poses]
        changed = True
        while len(out) > target and changed:
            changed = False
            best = None
            best_score = float("inf")
            for idx in range(1, len(out) - 1):
                if required_idx(out, idx):
                    continue
                edge_body, edge_arm = self._seed_edge_margin(out[idx - 1], out[idx + 1])
                if edge_body < self.required_body_margin:
                    continue
                deviation = point_distance(
                    self._pose_xyz(out[idx]),
                    self._pose_xyz(self._interpolate_pose_between(out[idx - 1], out[idx + 1], 0.5)),
                )
                score = deviation - 0.05 * edge_body - 0.01 * min(edge_arm, self.refine_clearance_target)
                if score < best_score:
                    best = idx
                    best_score = score
            if best is not None:
                del out[best]
                info["reference_keyframes_pruned"] += 1
                changed = True
        info["reference_keyframes_after_compression"] = len(out)
        return out, info

    def _trim_stale_terminal_reference_tail(self, poses, standoff_pose):
        if len(poses) < 2:
            return list(poses), 0
        out = list(poses)
        pruned = 0
        while len(out) > 1:
            dist = point_distance(self._pose_xyz(out[-1]), self._pose_xyz(standoff_pose))
            if dist <= 0.05 or dist >= 0.35:
                break
            out.pop()
            pruned += 1
        return out, pruned

    def _prepare_transit_reference(self, base_poses, standoff_pose, reference_info):
        self._publish_planning_state("prepare_reference_window")
        poses = self._local_refinement_window(base_poses)
        poses, terminal_tail_pruned = self._trim_stale_terminal_reference_tail(poses, standoff_pose)
        reference_info["terminal_tail_pruned"] = terminal_tail_pruned
        self._publish_planning_state("prepare_reference_approach")
        poses = self._insert_standoff_approach_if_needed(poses, standoff_pose)
        poses = self._append_standoff_if_needed(poses, standoff_pose)
        slot_info = {
            "slot_keyframes_inserted": 0,
            "slot_poses_projected": 0,
            "slot_keyframe_constraints": [],
        }
        if self.enable_reference_safety_repair:
            self._publish_planning_state("prepare_reference_slot_keyframes")
            poses, slot_info = self._repair_reference_with_slot_keyframes(poses)
            reference_info.update(slot_info)
            self._publish_planning_state("prepare_reference_edges")
            poses, edge_info = self._densify_reference_edges(poses)
        else:
            edge_info = self._measure_reference_edges(poses)
        reference_info.update(edge_info)
        slot_info.update(edge_info)
        self._publish_planning_state("prepare_reference_compress")
        poses, keyframe_info = self._compress_reference_keyframes(poses)
        reference_info.update(keyframe_info)
        slot_info.update(keyframe_info)
        return poses, reference_info, slot_info

    def _reference_edges_unresolved(self, reference_info):
        edge_margin_low = (
            float(reference_info.get("reference_edge_min_body_margin", float("inf"))) < self.required_body_margin
            or float(reference_info.get("reference_edge_min_arm_margin", float("inf"))) < self.required_arm_margin
        )
        return (
            int(reference_info.get("reference_edge_unresolved", 0)) > 0
            or (int(reference_info.get("reference_edge_splits", 0)) >= 24 and edge_margin_low)
        )

    def _global_crossing_x_for_slot(self, slot, fallback_x):
        if self.global_path is None or not self.global_path.poses:
            return fallback_x
        best_x = fallback_x
        best_dist = float("inf")
        for pose in self._path_poses(self.global_path):
            x, y, _ = self._pose_xyz(pose)
            if x < slot["x_min"] - 0.60 or x > slot["x_max"] + 0.60:
                continue
            dist = abs(y - slot["y"])
            if dist < best_dist:
                best_dist = dist
                best_x = x
        return best_x

    def _best_slot_x(self, slot, base_x, yaw):
        candidates = []
        for dx in (0.0, -0.25, 0.25, -0.50, 0.50, -0.85, 0.85, -1.20, 1.20):
            x = min(max(base_x + dx, slot["x_min"] + self.body_radius), slot["x_max"] - self.body_radius)
            if all(abs(x - old) > 1e-3 for old in candidates):
                candidates.append(x)
        best_x = base_x
        best_score = -float("inf")
        for x in candidates:
            margins = []
            for y in (slot["y"] - slot["half_y"], slot["y"], slot["y"] + slot["half_y"]):
                pose = self._make_pose(x, y, slot["z_target"], yaw)
                body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
                arm_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=7)
                margins.append(min(body_margin, arm_margin))
            score = min(margins) - 0.02 * abs(x - base_x)
            if score > best_score:
                best_score = score
                best_x = x
        return best_x

    def _build_slot_skeleton(self, standoff_pose):
        start_pose = self._copy_pose(self.current_pose)
        sx, sy, sz = self._pose_xyz(start_pose)
        gx, gy, gz = self._pose_xyz(standoff_pose)
        route_yaw = math.atan2(gy - sy, gx - sx)
        route = [start_pose]
        vx = gx - sx
        vy = gy - sy
        denom = max(vx * vx + vy * vy, 1e-6)
        ordered_slots = []
        for slot in self._slot_constraints():
            base_x = self._global_crossing_x_for_slot(slot, sx)
            t = ((base_x - sx) * vx + (slot["y"] - sy) * vy) / denom
            if -0.05 <= t <= 1.05:
                ordered_slots.append((t, slot, base_x))
        ordered_slots.sort(key=lambda item: item[0])
        for _, slot, base_x in ordered_slots:
            x = self._best_slot_x(slot, base_x, route_yaw)
            guard = slot["half_y"] + max(0.12, self.body_radius + self.body_clearance)
            direction = 1.0 if gy >= sy else -1.0
            for y in (slot["y"] - direction * guard, slot["y"], slot["y"] + direction * guard):
                pose = self._make_pose(x, y, slot["z_target"], route_yaw)
                if point_distance(self._pose_xyz(route[-1]), self._pose_xyz(pose)) >= 0.05:
                    route.append(pose)
        route = self._insert_standoff_approach_if_needed(route, standoff_pose)
        route = self._append_standoff_if_needed(route, standoff_pose)
        info = {
            "ego_reference_points": 0,
            "unsafe_ego_points": 0,
            "global_repair_points": 0,
            "global_repair_used": True,
            "global_repair_reason": "independent_slot_skeleton",
            "independent_skeleton_used": True,
            "independent_skeleton_points": len(route),
            "first_unsafe_ego_idx": -1,
            "first_unsafe_ego_margin": float("inf"),
            "first_unsafe_ego_obstacle": "none",
            "first_unsafe_ego_pose": (0.0, 0.0, 0.0),
            "first_unsafe_ego_point": (0.0, 0.0, 0.0),
        }
        return route, info

    def _skeleton_key(self, xyz):
        return (
            int(round(float(xyz[0]) / self.skeleton_xy_resolution)),
            int(round(float(xyz[1]) / self.skeleton_xy_resolution)),
            int(round(float(xyz[2]) / self.skeleton_z_resolution)),
        )

    def _skeleton_point(self, key):
        return (
            key[0] * self.skeleton_xy_resolution,
            key[1] * self.skeleton_xy_resolution,
            key[2] * self.skeleton_z_resolution,
        )

    def _skeleton_heuristic(self, key, goal_key):
        return math.sqrt(
            ((key[0] - goal_key[0]) * self.skeleton_xy_resolution) ** 2
            + ((key[1] - goal_key[1]) * self.skeleton_xy_resolution) ** 2
            + ((key[2] - goal_key[2]) * self.skeleton_z_resolution) ** 2
        )

    def _skeleton_safe_point(self, xyz):
        margin, _, _ = self._body_point_margin(xyz)
        return margin >= self.required_body_margin

    def _skeleton_safe_segment(self, a, b):
        dist = point_distance(a, b)
        step = max(0.02, min(self.skeleton_xy_resolution, self.skeleton_z_resolution) * 0.5)
        samples = max(2, int(math.ceil(dist / step)) + 1)
        for idx in range(samples):
            ratio = idx / float(samples - 1)
            point = (
                a[0] + (b[0] - a[0]) * ratio,
                a[1] + (b[1] - a[1]) * ratio,
                a[2] + (b[2] - a[2]) * ratio,
            )
            if not self._skeleton_safe_point(point):
                return False
        return True

    def _nearest_skeleton_safe_key(self, center_key, bounds, max_shell=8):
        if self._skeleton_safe_point(self._skeleton_point(center_key)):
            return center_key
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        best = None
        best_dist2 = float("inf")
        for shell in range(1, max_shell + 1):
            for dx in range(-shell, shell + 1):
                for dy in range(-shell, shell + 1):
                    for dz in range(-shell, shell + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != shell:
                            continue
                        key = (center_key[0] + dx, center_key[1] + dy, center_key[2] + dz)
                        if key[0] < x_min or key[0] > x_max or key[1] < y_min or key[1] > y_max:
                            continue
                        if key[2] < z_min or key[2] > z_max:
                            continue
                        if not self._skeleton_safe_point(self._skeleton_point(key)):
                            continue
                        dist2 = dx * dx + dy * dy + dz * dz
                        if dist2 < best_dist2:
                            best = key
                            best_dist2 = dist2
            if best is not None:
                return best
        return None

    def _skeleton_segment_poses(self, a, b):
        ax, ay, az, ayaw = a
        bx, by, bz, byaw = b
        dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
        steps = max(2, int(math.ceil(dist / max(self.skeleton_path_step, 1e-3))) + 1)
        poses = []
        for idx in range(steps):
            ratio = idx / float(steps - 1)
            poses.append(
                self._make_pose(
                    ax + (bx - ax) * ratio,
                    ay + (by - ay) * ratio,
                    az + (bz - az) * ratio,
                    self._interp_yaw(ayaw, byaw, ratio),
                )
            )
        return poses

    def _skeleton_route_from_points(self, points):
        poses = []
        for idx in range(len(points) - 1):
            segment = self._skeleton_segment_poses(points[idx], points[idx + 1])
            if poses:
                segment = segment[1:]
            poses.extend(segment)
        return poses

    def _build_body_astar_skeleton(self, standoff_pose):
        info = {
            "ego_reference_points": 0,
            "unsafe_ego_points": 0,
            "global_repair_points": 0,
            "global_repair_used": True,
            "global_repair_reason": "body_free_astar_skeleton",
            "independent_skeleton_used": True,
            "independent_skeleton_points": 0,
            "body_skeleton_used": True,
            "body_skeleton_status": "not_started",
            "body_skeleton_expanded": 0,
            "body_skeleton_raw_points": 0,
            "body_skeleton_start_margin": float("-inf"),
            "body_skeleton_goal_margin": float("-inf"),
            "first_unsafe_ego_idx": -1,
            "first_unsafe_ego_margin": float("inf"),
            "first_unsafe_ego_obstacle": "none",
            "first_unsafe_ego_pose": (0.0, 0.0, 0.0),
            "first_unsafe_ego_point": (0.0, 0.0, 0.0),
        }
        if self.current_pose is None:
            info["body_skeleton_status"] = "missing_current_pose"
            return None, info

        previous_cloud = self.cloud_collision_enabled
        self.cloud_collision_enabled = self.static_cloud_in_optimization
        try:
            start_pose = self._copy_pose(self.current_pose)
            sx, sy, sz = self._pose_xyz(start_pose)
            gx, gy, gz = self._pose_xyz(standoff_pose)
            syaw = self._pose_yaw(start_pose)
            gyaw = self._pose_yaw(standoff_pose)
            planning_sz = max(sz, self.skeleton_z_min)
            planning_gz = min(max(gz, self.skeleton_z_min), self.skeleton_z_max)
            start_key = self._skeleton_key((sx, sy, planning_sz))
            goal_key = self._skeleton_key((gx, gy, planning_gz))
            x_pad = int(math.ceil(self.skeleton_xy_padding / self.skeleton_xy_resolution))
            y_pad = int(math.ceil(self.skeleton_xy_padding / self.skeleton_xy_resolution))
            z_min = int(math.floor(self.skeleton_z_min / self.skeleton_z_resolution))
            z_max = int(math.ceil(self.skeleton_z_max / self.skeleton_z_resolution))
            bounds = (
                min(start_key[0], goal_key[0]) - x_pad,
                max(start_key[0], goal_key[0]) + x_pad,
                min(start_key[1], goal_key[1]) - y_pad,
                max(start_key[1], goal_key[1]) + y_pad,
                z_min,
                z_max,
            )

            start_margin, _, _ = self._body_point_margin(self._skeleton_point(start_key))
            goal_margin, _, _ = self._body_point_margin(self._skeleton_point(goal_key))
            info["body_skeleton_start_margin"] = start_margin
            info["body_skeleton_goal_margin"] = goal_margin
            start_key = self._nearest_skeleton_safe_key(start_key, bounds, max_shell=16)
            goal_key = self._nearest_skeleton_safe_key(goal_key, bounds, max_shell=16)
            if start_key is None:
                info["body_skeleton_status"] = "start_not_safe"
                return None, info
            if goal_key is None:
                info["body_skeleton_status"] = "goal_not_safe"
                return None, info

            neighbors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        cost = math.sqrt(
                            (dx * self.skeleton_xy_resolution) ** 2
                            + (dy * self.skeleton_xy_resolution) ** 2
                            + (dz * self.skeleton_z_resolution) ** 2
                        )
                        neighbors.append((dx, dy, dz, cost))

            open_heap = [(self.skeleton_heuristic_weight * self._skeleton_heuristic(start_key, goal_key), 0.0, start_key)]
            came_from = {}
            g_score = {start_key: 0.0}
            closed = set()
            safe_cache = {}
            segment_cache = {}
            clearance_cache = {}
            x_min, x_max, y_min, y_max, _, _ = bounds

            while open_heap and len(closed) < self.skeleton_max_expansions:
                _, current_g, current = heapq.heappop(open_heap)
                if current in closed:
                    continue
                if current == goal_key:
                    keys = [current]
                    while keys[-1] != start_key:
                        keys.append(came_from[keys[-1]])
                    keys.reverse()
                    points = []
                    for idx, key in enumerate(keys):
                        x, y, z = self._skeleton_point(key)
                        ratio = idx / float(max(len(keys) - 1, 1))
                        points.append((x, y, z, self._interp_yaw(syaw, gyaw, ratio)))
                    points[0] = (sx, sy, planning_sz, syaw)
                    points[-1] = (gx, gy, planning_gz, gyaw)
                    route = self._skeleton_route_from_points(points)
                    info["body_skeleton_status"] = "ready"
                    info["body_skeleton_expanded"] = len(closed)
                    info["body_skeleton_raw_points"] = len(points)
                    info["independent_skeleton_points"] = len(route)
                    return route, info

                closed.add(current)
                for dx, dy, dz, step_cost in neighbors:
                    nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if nb in closed:
                        continue
                    if nb[0] < x_min or nb[0] > x_max or nb[1] < y_min or nb[1] > y_max:
                        continue
                    if nb[2] < z_min or nb[2] > z_max:
                        continue
                    if nb not in safe_cache:
                        safe_cache[nb] = self._skeleton_safe_point(self._skeleton_point(nb))
                    if not safe_cache[nb]:
                        continue
                    edge_key = (current, nb)
                    if edge_key not in segment_cache:
                        segment_cache[edge_key] = self._skeleton_safe_segment(
                            self._skeleton_point(current),
                            self._skeleton_point(nb),
                        )
                    if not segment_cache[edge_key]:
                        continue
                    if nb not in clearance_cache:
                        margin, _, _ = self._body_point_margin(self._skeleton_point(nb))
                        deficit = max(0.0, self.refine_clearance_target - margin) / max(self.refine_clearance_target, 1e-3)
                        clearance_cache[nb] = 1.0 + 2.0 * deficit * deficit
                    tentative = current_g + step_cost * clearance_cache[nb]
                    if tentative >= g_score.get(nb, float("inf")):
                        continue
                    came_from[nb] = current
                    g_score[nb] = tentative
                    heapq.heappush(
                        open_heap,
                        (
                            tentative + self.skeleton_heuristic_weight * self._skeleton_heuristic(nb, goal_key),
                            tentative,
                            nb,
                        ),
                    )

            info["body_skeleton_status"] = "blocked_or_budget"
            info["body_skeleton_expanded"] = len(closed)
            return None, info
        finally:
            self.cloud_collision_enabled = previous_cloud

    def _insert_standoff_approach_if_needed(self, poses, standoff_pose):
        if not poses:
            return poses
        last = poses[-1]
        lx, ly, lz = self._pose_xyz(last)
        sx, sy, sz = self._pose_xyz(standoff_pose)
        dist = point_distance((lx, ly, lz), (sx, sy, sz))
        if dist < 0.50:
            edge_body, edge_arm = self._seed_edge_margin(last, standoff_pose)
            if edge_body >= self.required_body_margin and edge_arm >= self.required_arm_margin:
                return poses
        dx = lx - sx
        dy = ly - sy
        dz = lz - sz
        norm = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
        ux = dx / norm
        uy = dy / norm
        px = -uy
        py = ux
        base_step = min(max(self.standoff_approach_distance, 1.05), max(0.35, 0.60 * dist))
        max_step = min(max(self.standoff_approach_distance, 1.20), max(0.45, 0.85 * dist))
        step_values = sorted(
            set(
                [
                    min(self.standoff_approach_distance, 0.35),
                    min(self.standoff_approach_distance, 0.50),
                    min(self.standoff_approach_distance, 0.65),
                    min(max_step, 0.90),
                    min(max_step, 1.05),
                    base_step,
                    max_step,
                ]
            )
        )
        lateral_values = [0.0, -0.20, 0.20, -0.40, 0.40, -0.65, 0.65]
        z_values = [
            sz + self.standoff_approach_z_lift,
            max(sz + self.standoff_approach_z_lift, min(max(lz, sz), sz + 0.35)),
            sz + self.standoff_approach_z_lift + 0.12,
            sz + self.standoff_approach_z_lift - 0.08,
            sz + self.standoff_approach_z_lift + 0.22,
            sz - 0.18,
            sz - 0.10,
            sz + 0.30,
            max(0.90, sz + 0.02),
        ]
        candidates = []
        for step in step_values:
            for lateral in lateral_values:
                for z in z_values:
                    approach_x = sx + ux * step + px * lateral
                    approach_y = sy + uy * step + py * lateral
                    approach_z = max(0.90, min(2.05, z))
                    approach_yaw = math.atan2(sy - approach_y, sx - approach_x)
                    approach_pose = self._make_pose(approach_x, approach_y, approach_z, approach_yaw)
                    point_margin, _, _ = self._body_point_margin(self._pose_xyz(approach_pose))
                    edge_body, edge_arm = self._seed_edge_margin(approach_pose, standoff_pose)
                    last_body, last_arm = self._seed_edge_margin(last, approach_pose)
                    min_body = min(point_margin, edge_body, last_body)
                    min_arm = min(edge_arm, last_arm)
                    score = min(min_body, min_arm) + 0.35 * point_margin - 0.05 * abs(lateral)
                    if edge_body < self.required_body_margin or edge_arm < self.required_arm_margin:
                        score -= 10.0
                    if last_body < self.required_body_margin or last_arm < self.required_arm_margin:
                        score -= 10.0
                    candidates.append(
                        {
                            "pose": approach_pose,
                            "step": step,
                            "lateral": lateral,
                            "dz": approach_z - sz,
                            "score": score,
                            "point_margin": point_margin,
                            "last_body": last_body,
                            "last_arm": last_arm,
                            "standoff_body": edge_body,
                            "standoff_arm": edge_arm,
                            "single_safe": (
                                min_body >= self.required_body_margin
                                and min_arm >= self.required_arm_margin
                            ),
                        }
                    )
        if not candidates:
            return poses

        best_single = max(candidates, key=lambda item: item["score"])
        safe_single = None
        safe_single_score = -float("inf")
        for candidate in candidates:
            if not candidate["single_safe"]:
                continue
            if candidate["score"] > safe_single_score:
                safe_single = candidate
                safe_single_score = candidate["score"]
        best_pair = None
        best_pair_score = -float("inf")
        # Build an ordered two-point approach ladder: farther point first,
        # nearer point second. This keeps swept edges short near the rear
        # standoff rails without changing the final safety constraints.
        pair_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:18]
        for far in pair_candidates:
            for near in pair_candidates:
                if far["step"] <= near["step"] + 0.12:
                    continue
                edge0_body = far["last_body"]
                edge0_arm = far["last_arm"]
                edge1_body, edge1_arm = self._seed_edge_margin(far["pose"], near["pose"])
                edge2_body = near["standoff_body"]
                edge2_arm = near["standoff_arm"]
                min_body = min(edge0_body, edge1_body, edge2_body, far["point_margin"], near["point_margin"])
                min_arm = min(edge0_arm, edge1_arm, edge2_arm)
                pair_safe = min_body >= self.required_body_margin and min_arm >= self.required_arm_margin
                body_deficit = max(0.0, self.sequence_risk_body_margin - min_body) / max(self.sequence_risk_body_margin, 1e-3)
                arm_deficit = max(0.0, self.sequence_risk_arm_margin - min_arm) / max(self.sequence_risk_arm_margin, 1e-3)
                shape = abs(far["lateral"]) + abs(near["lateral"]) + 0.5 * abs(far["dz"] - near["dz"])
                score = (
                    2.0 * min(min_body, self.sequence_risk_body_margin)
                    + 1.5 * min(min_arm, self.sequence_risk_arm_margin)
                    - 8.0 * body_deficit * body_deficit
                    - 8.0 * arm_deficit * arm_deficit
                    - 0.06 * shape
                )
                if pair_safe and score > best_pair_score:
                    best_pair_score = score
                    best_pair = (far, near)

        if best_pair is None:
            if safe_single is not None:
                return list(poses) + [safe_single["pose"]]
            return poses
        return list(poses) + [best_pair[0]["pose"], best_pair[1]["pose"]]

    def _transition_failure_wp(self, status):
        prefixes = (
            "WHOLE_BODY_SEQUENCE_TRANSITION_FAILED_WP_",
            "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_",
            "WHOLE_BODY_REFINEMENT_FAILED_WP_",
        )
        for prefix in prefixes:
            if not status.startswith(prefix):
                continue
            try:
                return int(status[len(prefix):])
            except ValueError:
                return None
        return None

    def _insert_transition_repair_waypoints(self, poses, status):
        failure_idx = self._transition_failure_wp(status)
        if failure_idx is None or len(poses) < 2:
            return list(poses), 0
        right = min(max(failure_idx, 1), len(poses) - 1)
        out = []
        inserted = 0
        for idx, pose in enumerate(poses):
            if idx == right:
                prev_pose = poses[idx - 1]
                ladder = self._slot_projected_edge_ladder(prev_pose, pose)
                if ladder:
                    out.extend(ladder)
                    inserted += len(ladder)
                else:
                    detour = self._transition_detour_pose(prev_pose, pose)
                    if detour is not None:
                        out.append(detour)
                        inserted += 1
                    else:
                        first = self._interpolate_pose_between(prev_pose, pose, 1.0 / 3.0)
                        second = self._interpolate_pose_between(prev_pose, pose, 2.0 / 3.0)
                        first, _, _ = self._project_pose_to_slot_window(first)
                        second, _, _ = self._project_pose_to_slot_window(second)
                        out.append(first)
                        out.append(second)
                        inserted += 2
            out.append(self._copy_pose(pose))
        return out, inserted

    def _transition_repair_window_indices(self, count, failure_idx, inserted):
        if count <= 0 or failure_idx is None:
            return set()
        # Repair failures are reported at the right endpoint of the failed
        # transition. Inserted knots appear before that endpoint, so the repair
        # window must include the original failed endpoint, the inserted knots,
        # and a small downstream tail. Centering only at failure_idx+inserted
        # misses the actual failed layer when several slot knots are inserted.
        failure_idx = min(max(int(failure_idx), 0), count - 1)
        inserted = max(0, int(inserted))
        backward = 3 + min(2, inserted // 3)
        downstream = 6
        start = max(0, failure_idx - backward)
        end = min(count, failure_idx + inserted + downstream + 1)
        return set(range(start, end))

    def _insert_swept_repair_waypoints(self, base_poses, validated_poses, validated_shapes, swept_segment):
        if (
            swept_segment < 0
            or swept_segment + 1 >= len(validated_poses)
            or swept_segment + 1 >= len(validated_shapes)
            or len(base_poses) < 2
        ):
            return list(base_poses), 0, -1
        fail_mid = self._pose_xyz(self._interpolate_pose_between(validated_poses[swept_segment], validated_poses[swept_segment + 1], 0.5))
        best_idx = 1
        best_dist = float("inf")
        for idx in range(1, len(base_poses)):
            seg_mid = self._pose_xyz(self._interpolate_pose_between(base_poses[idx - 1], base_poses[idx], 0.5))
            dist = point_distance(fail_mid, seg_mid)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        out = []
        ladder = self._slot_projected_edge_ladder(validated_poses[swept_segment], validated_poses[swept_segment + 1])
        detour = None
        if not ladder:
            detour = self._swept_detour_pose(
                validated_poses[swept_segment],
                validated_poses[swept_segment + 1],
                validated_shapes[swept_segment],
                validated_shapes[swept_segment + 1],
            )
            if detour is None:
                detour = self._transition_detour_pose(validated_poses[swept_segment], validated_poses[swept_segment + 1])
        for idx, pose in enumerate(base_poses):
            if idx == best_idx:
                prev_pose = base_poses[idx - 1]
                if ladder:
                    out.extend(ladder)
                elif detour is not None:
                    out.append(detour)
                else:
                    first = self._interpolate_pose_between(prev_pose, pose, 1.0 / 3.0)
                    second = self._interpolate_pose_between(prev_pose, pose, 2.0 / 3.0)
                    first, _, _ = self._project_pose_to_slot_window(first)
                    second, _, _ = self._project_pose_to_slot_window(second)
                    out.append(first)
                    out.append(second)
            out.append(self._copy_pose(pose))
        if ladder:
            return out, len(ladder), best_idx
        return out, 1 if detour is not None else 2, best_idx

    def _search_safe_transit_shape(self, pose, preferred_theta, preferred_phi):
        if self._arm_safe(pose, preferred_theta, preferred_phi):
            return preferred_theta, preferred_phi
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)
        theta_values = []
        theta = float(self.arm_params["theta_min_deg"])
        while theta <= float(self.arm_params["theta_max_deg"]) + 1e-9:
            theta_values.append(theta)
            theta += theta_step
        phi_values = []
        phi = float(self.arm_params["phi_min_deg"])
        while phi <= float(self.arm_params["phi_max_deg"]) + 1e-9:
            phi_values.append(phi)
            phi += phi_step
        best = None
        best_cost = float("inf")
        for theta in theta_values:
            for phi in phi_values:
                if not self._arm_safe(pose, theta, phi):
                    continue
                compact_cost = abs(theta - self.compact_theta) + 0.2 * abs(self._angle_delta(phi, self.compact_phi))
                continuity_cost = abs(theta - preferred_theta) + 0.25 * abs(self._angle_delta(phi, preferred_phi))
                cost = continuity_cost + 0.2 * compact_cost
                if cost < best_cost:
                    best = (theta, phi)
                    best_cost = cost
        return best

    def _assign_transit_shapes(self, poses):
        waypoints = []
        last_theta = self.current_theta_phi[0]
        last_phi = self.current_theta_phi[1]
        for idx, pose in enumerate(poses):
            if not self._body_safe(pose):
                return None, "BODY_COLLISION_TRANSIT_WP_%d" % idx
            if idx == 0:
                preferred_theta, preferred_phi = last_theta, last_phi
            else:
                preferred_theta, preferred_phi = self.compact_theta, self.compact_phi
            shape = self._search_safe_transit_shape(pose, preferred_theta, preferred_phi)
            if shape is None:
                return None, "ARM_BACKBONE_COLLISION_TRANSIT_WP_%d" % idx
            theta, phi = shape
            waypoints.append((theta, phi))
            last_theta, last_phi = theta, phi
        return waypoints, "OK"

    def _path_yaw_at(self, poses, idx):
        if len(poses) < 2:
            return self._pose_yaw(poses[idx])
        if idx < len(poses) - 1:
            a = self._pose_xyz(poses[idx])
            b = self._pose_xyz(poses[idx + 1])
        else:
            a = self._pose_xyz(poses[idx - 1])
            b = self._pose_xyz(poses[idx])
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _offset_pose(self, pose, path_yaw, lateral, dz, yaw_offset_deg):
        base = self._pose_xyz(pose)
        left = (-math.sin(path_yaw), math.cos(path_yaw))
        yaw = path_yaw + math.radians(yaw_offset_deg)
        return self._make_pose(base[0] + left[0] * lateral, base[1] + left[1] * lateral, base[2] + dz, yaw)

    def _arm_margin_detail(self, pose, theta, phi, sample_limit=None):
        pose_key = (
            round(float(pose.position.x), 3),
            round(float(pose.position.y), 3),
            round(float(pose.position.z), 3),
            round(float(self._pose_yaw(pose)), 3),
            round(float(theta), 2),
            round(float(phi), 2),
            int(sample_limit) if sample_limit is not None else -1,
            bool(self.cloud_collision_enabled),
        )
        cached = self._arm_margin_cache.get(pose_key)
        if cached is not None:
            return cached
        threshold = self.arm_params["radius"] + self.arm_clearance
        best = float("inf")
        best_name = "none"
        best_point = (0.0, 0.0, 0.0)
        points = world_arm_points(pose, theta, phi, self.arm_params)
        if sample_limit is not None and len(points) > sample_limit:
            points = [points[int(round(i * (len(points) - 1) / float(max(sample_limit - 1, 1))))] for i in range(sample_limit)]
        for point in points:
            margin, name, near = self._point_margin(point, threshold, False)
            if margin < best:
                best = margin
                best_name = name
                best_point = near
        result = (best, best_name, best_point)
        if len(self._arm_margin_cache) < 300000:
            self._arm_margin_cache[pose_key] = result
        return result

    def _arm_margin(self, pose, theta, phi, sample_limit=None):
        margin, _, _ = self._arm_margin_detail(pose, theta, phi, sample_limit)
        return margin

    def _morphology_activation(self, home_margin, is_goal_pose=False):
        if is_goal_pose:
            return 1.0
        target = self.morphology_activation_margin
        if home_margin >= target:
            return 0.0
        return min(1.0, max(0.0, (target - home_margin) / self.morphology_activation_band))

    def _compact_shape_distance(self, theta, phi):
        return abs(theta - self.compact_theta) + 0.25 * abs(self._angle_delta(phi, self.compact_phi))

    def _shape_distance(self, a, b):
        return abs(a[0] - b[0]) + 0.25 * abs(self._angle_delta(a[1], b[1]))

    def _blend_shape(self, a, b, ratio):
        ratio = min(1.0, max(0.0, float(ratio)))
        theta = a[0] + (b[0] - a[0]) * ratio
        phi = self._normalize_angle_deg(a[1] + self._angle_delta(a[1], b[1]) * ratio)
        return (theta, phi)

    def _release_reference_shape(self, stable, reach, release):
        release = min(1.0, max(0.0, float(release)))
        phi_delta = abs(self._angle_delta(stable[1], reach[1]))
        if phi_delta < 70.0:
            return self._blend_shape(stable, reach, release)

        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        low_theta = min(max(theta_min, self.compact_theta), theta_max)
        low_stable = (low_theta, stable[1])
        low_reach = (low_theta, reach[1])

        lower_end = 0.30
        rotate_end = 0.68
        if release <= lower_end:
            return self._blend_shape(stable, low_stable, release / max(lower_end, 1e-3))
        if release <= rotate_end:
            return self._blend_shape(
                low_stable,
                low_reach,
                (release - lower_end) / max(rotate_end - lower_end, 1e-3),
            )
        return self._blend_shape(
            low_reach,
            reach,
            (release - rotate_end) / max(1.0 - rotate_end, 1e-3),
        )

    def _release_bridge_shape_candidates(self, stable, reach, q_ref):
        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        low_theta = min(max(theta_min, self.compact_theta), theta_max)
        active_low_theta = min(max(self.morphology_min_active_theta, theta_min), theta_max)
        seeds = [
            stable,
            q_ref,
            reach,
            (low_theta, stable[1]),
            (low_theta, q_ref[1]),
            (low_theta, reach[1]),
            (active_low_theta, q_ref[1]),
            (active_low_theta, reach[1]),
        ]
        for ratio in (0.25, 0.50, 0.75):
            phi = self._normalize_angle_deg(stable[1] + self._angle_delta(stable[1], reach[1]) * ratio)
            seeds.append((low_theta, phi))
            seeds.append((active_low_theta, phi))
        for ratio in (0.35, 0.70):
            seeds.append(self._blend_shape(stable, q_ref, ratio))
            seeds.append(self._blend_shape(q_ref, reach, ratio))
        out = []
        seen = set()
        for theta, phi in seeds:
            theta = min(max(float(theta), theta_min), theta_max)
            phi = self._normalize_angle_deg(float(phi))
            key = (round(theta, 3), round(phi, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append((theta, phi))
        return out

    def _smoothstep(self, value):
        value = min(1.0, max(0.0, float(value)))
        return value * value * (3.0 - 2.0 * value)

    def _stable_reference_shape(self, pose):
        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        theta_values = [
            theta_min,
            theta_min + 0.15 * (theta_max - theta_min),
            theta_min + 0.30 * (theta_max - theta_min),
            theta_min + 0.50 * (theta_max - theta_min),
            theta_min + 0.70 * (theta_max - theta_min),
            self.compact_theta,
        ]
        phi_values = [
            self.compact_phi,
            0.0,
            45.0,
            -45.0,
            90.0,
            -90.0,
            180.0,
            -180.0,
        ]
        seeds = [(theta, phi) for theta in theta_values for phi in phi_values]
        best = (theta_min, self.compact_phi)
        best_cost = float("inf")
        seen = set()
        for theta, phi in seeds:
            theta = min(max(theta, theta_min), theta_max)
            phi = self._normalize_angle_deg(phi)
            key = (round(theta, 3), round(phi, 3))
            if key in seen:
                continue
            seen.add(key)
            arm_margin = self._arm_margin(pose, theta, phi, sample_limit=5)
            if arm_margin < self.required_arm_margin:
                continue
            coupling = self._coupling_metrics(pose, theta, phi)
            _, lateral_extent = self._arm_morphology_metrics(pose, theta, phi)
            clearance_deficit = max(0.0, self.sequence_risk_arm_margin - arm_margin) / max(self.sequence_risk_arm_margin, 1e-3)
            cost = (
                28.0 * coupling["com_offset"]
                + 1.8 * coupling["gravity_torque"]
                + 1.8 * lateral_extent
                + 5.0 * clearance_deficit * clearance_deficit
            )
            if cost < best_cost:
                best_cost = cost
                best = (theta, phi)
        return best

    def _risk_segment_key_for_pose(self, pose):
        body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
        slot = self._slot_constraint_for_pose(pose)
        compact_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=5)
        body_target = max(self.sequence_risk_body_margin, self.required_body_margin + 0.04)
        arm_target = max(self.sequence_risk_arm_margin, self.required_arm_margin + 0.04)
        if slot is not None:
            return slot["id"], body_margin, compact_margin
        if body_margin < body_target or compact_margin < arm_target:
            return "risk", body_margin, compact_margin
        return None, body_margin, compact_margin

    def _segment_shape_seed_candidates(self, poses, goal_shape):
        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        active_low_theta = min(max(self.morphology_min_active_theta, theta_min), theta_max)
        seeds = [
            (theta_min, self.compact_phi),
            (active_low_theta, self.compact_phi),
            (self.compact_theta, self.compact_phi),
            goal_shape,
        ]
        sample_poses = poses
        if len(sample_poses) > 5:
            sample_poses = [
                poses[0],
                poses[len(poses) // 4],
                poses[len(poses) // 2],
                poses[(3 * len(poses)) // 4],
                poses[-1],
            ]
        for pose in sample_poses:
            stable = self._stable_reference_shape(pose)
            seeds.extend(
                [
                    stable,
                    (theta_min, stable[1]),
                    (active_low_theta, stable[1]),
                    (min(theta_max, max(active_low_theta, stable[0])), stable[1]),
                ]
            )
            slot = self._slot_constraint_for_pose(pose)
            if slot is not None:
                seeds.extend(self._slot_shape_seeds(pose))
        out = []
        seen = set()
        for theta, phi in seeds:
            theta = min(max(float(theta), theta_min), theta_max)
            phi = self._normalize_angle_deg(float(phi))
            key = (round(theta, 3), round(phi, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append((theta, phi))
        return out

    def _evaluate_hold_shape_for_segment(self, poses, shape):
        theta, phi = shape
        min_arm = float("inf")
        max_com = 0.0
        max_torque = 0.0
        max_lateral = 0.0
        arm_deficit = 0.0
        for pose in poses:
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            if body_margin < self.required_body_margin:
                return None
            arm_margin = self._arm_margin(pose, theta, phi, sample_limit=7)
            min_arm = min(min_arm, arm_margin)
            if arm_margin < self.required_arm_margin:
                return None
            coupling = self._coupling_metrics(pose, theta, phi)
            if coupling["com_offset"] > self.hard_com_limit or coupling["gravity_torque"] > self.tau_limit:
                return None
            _, lateral_extent = self._arm_morphology_metrics(pose, theta, phi)
            max_com = max(max_com, coupling["com_offset"])
            max_torque = max(max_torque, coupling["gravity_torque"])
            max_lateral = max(max_lateral, lateral_extent)
            deficit = max(0.0, self.sequence_risk_arm_margin - arm_margin) / max(self.sequence_risk_arm_margin, 1e-3)
            arm_deficit += deficit * deficit
        compact_dist = self._compact_shape_distance(theta, phi)
        score = (
            40.0 * max_com
            + 2.0 * max_torque
            + 2.0 * max_lateral
            + 6.0 * arm_deficit
            + 0.010 * compact_dist
            - 1.2 * min(min_arm, self.sequence_risk_arm_margin)
        )
        return score, min_arm, max_com, max_torque

    def _build_passage_shape_schedule(self, ego_poses, goal_theta, goal_phi):
        schedule = [None for _ in ego_poses]
        info = {
            "passage_segment_count": 0,
            "passage_hold_wp_count": 0,
            "passage_hold_shape_count": 0,
            "passage_hold_min_arm_margin": float("inf"),
        }
        if not ego_poses:
            return schedule, info
        keys = []
        for pose in ego_poses:
            key, _, _ = self._risk_segment_key_for_pose(pose)
            keys.append(key)
        segments = []
        start = None
        current_key = None
        for idx, key in enumerate(keys + [None]):
            if key is not None and start is None:
                start = idx
                current_key = key
                continue
            if start is not None and (key != current_key or key is None):
                end = idx - 1
                if end - start + 1 >= 2:
                    segments.append((start, end, current_key))
                start = idx if key is not None else None
                current_key = key
        info["passage_segment_count"] = len(segments)
        goal_shape = (goal_theta, goal_phi)
        for segment_id, (start, end, key) in enumerate(segments):
            poses = ego_poses[start : end + 1]
            best = None
            for shape in self._segment_shape_seed_candidates(poses, goal_shape):
                result = self._evaluate_hold_shape_for_segment(poses, shape)
                if result is None:
                    continue
                score, min_arm, max_com, max_torque = result
                if best is None or score < best["score"]:
                    best = {
                        "shape": shape,
                        "score": score,
                        "min_arm": min_arm,
                        "max_com": max_com,
                        "max_torque": max_torque,
                    }
            if best is None:
                continue
            hold_shape = best["shape"]
            for idx in range(start, end + 1):
                schedule[idx] = {
                    "segment_id": segment_id,
                    "segment_key": key,
                    "segment_start": start,
                    "segment_end": end,
                    "hold_shape": hold_shape,
                    "hold_min_arm_margin": best["min_arm"],
                    "hold_score": best["score"],
                }
            # Give adjacent low-risk buffers access to the same hold shape so
            # phi/theta can be aligned before entering and after leaving.
            for idx in (start - 1, end + 1):
                if 0 <= idx < len(schedule) and schedule[idx] is None:
                    schedule[idx] = {
                        "segment_id": segment_id,
                        "segment_key": "%s_buffer" % key,
                        "segment_start": start,
                        "segment_end": end,
                        "hold_shape": hold_shape,
                        "hold_min_arm_margin": best["min_arm"],
                        "hold_score": best["score"],
                        "segment_buffer": True,
                    }
            info["passage_hold_shape_count"] += 1
            info["passage_hold_min_arm_margin"] = min(info["passage_hold_min_arm_margin"], best["min_arm"])
        info["passage_hold_wp_count"] = sum(1 for item in schedule if item is not None and not item.get("segment_buffer", False))
        if not math.isfinite(info["passage_hold_min_arm_margin"]):
            info["passage_hold_min_arm_margin"] = 0.0
        return schedule, info

    def _morphology_release_factor(self, ego_poses, idx, passage_risk, passage_slot="none"):
        count = max(len(ego_poses), 1)
        progress = float(idx) / float(max(count - 1, 1))
        start = min(0.95, max(0.0, self.morphology_release_start_fraction))
        progress_release = self._smoothstep((progress - start) / max(1.0 - start, 1e-3))
        target_point = tuple(self.target["pose"][:3])
        distance_to_target = point_distance(self._pose_xyz(ego_poses[idx]), target_point)
        distance_release = self._smoothstep(
            (self.morphology_release_target_distance - distance_to_target)
            / max(self.morphology_release_target_distance, 1e-3)
        )
        release = max(progress_release, distance_release)
        if passage_risk:
            if passage_slot != "none":
                release *= 0.35
            else:
                # Non-slot risk near the target is approach risk, not passage
                # risk. Keep enough release authority there so the arm can
                # pre-shape before the rear rails/standoff instead of trying to
                # change shape on the final transition.
                release *= max(0.35, distance_release)
        return min(1.0, max(0.0, release))

    def _passage_morphology_context(self, pose, body_margin, compact_arm_margin, idx, count, slot_constraint=None):
        if slot_constraint is None:
            slot_constraint = self._slot_constraint_for_pose(pose)
        body_target = max(self.sequence_risk_body_margin, self.required_body_margin + 0.04)
        arm_target = max(self.sequence_risk_arm_margin, self.required_arm_margin + 0.04)
        passage_risk = (
            slot_constraint is not None
            or body_margin < body_target
            or compact_arm_margin < arm_target
        )
        compact_safe = (
            body_margin >= self.required_body_margin + self.passage_compact_body_margin_buffer
            and compact_arm_margin >= self.required_arm_margin + self.passage_compact_arm_margin_buffer
        )
        return {
            "passage_risk": bool(passage_risk),
            "passage_slot": slot_constraint["id"] if slot_constraint is not None else "none",
            "passage_compact_safe": bool(compact_safe),
            "passage_compact_lock": False,
        }

    def _lookahead_morphology_activation(self, ego_poses, idx):
        if not ego_poses:
            return 0.0
        activation = 0.0
        target = max(self.morphology_activation_margin, self.required_arm_margin + 0.05, self.refine_clearance_target)
        band = max(self.morphology_activation_band, 1e-3)
        start = max(0, idx - 2)
        end = min(len(ego_poses), idx + 7)
        origin = self._pose_xyz(ego_poses[idx])
        travelled = 0.0
        last = origin
        for look_idx in range(start, end):
            pose = ego_poses[look_idx]
            point = self._pose_xyz(pose)
            if look_idx > idx:
                travelled += point_distance(last, point)
                last = point
                if travelled > 1.60:
                    break
            body_margin, _, _ = self._body_point_margin(point)
            home_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=5)
            risk_margin = min(body_margin, home_margin)
            if risk_margin < target:
                activation = max(activation, min(1.0, max(0.0, (target - risk_margin) / band)))
        return activation

    def _shape_candidates(self, previous, desired, morphology_activation=0.0):
        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        phi_min = float(self.arm_params["phi_min_deg"])
        phi_max = float(self.arm_params["phi_max_deg"])
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)

        theta_values = []
        theta = theta_min
        while theta <= theta_max + 1e-9:
            theta_values.append(theta)
            theta += theta_step
        for theta in [previous[0], desired[0], self.compact_theta, self.morphology_min_active_theta]:
            theta_values.append(theta)

        phi_values = []
        phi = phi_min
        while phi <= phi_max + 1e-9:
            phi_values.append(phi)
            phi += phi_step
        for phi in [previous[1], desired[1], self.compact_phi]:
            phi_values.append(phi)

        raw = []
        active_seeds = []
        if morphology_activation > 0.5:
            active_thetas = [
                self.morphology_min_active_theta,
                max(self.morphology_min_active_theta, 60.0),
                max(self.morphology_min_active_theta, 90.0),
                theta_max,
            ]
            active_phis = [
                previous[1],
                desired[1],
                self.compact_phi,
                -180.0,
                -135.0,
                -90.0,
                -45.0,
                0.0,
                45.0,
                90.0,
                135.0,
                180.0,
            ]
            for phi in active_phis:
                for theta in active_thetas:
                    active_seeds.append((theta, phi))

        for base_theta, base_phi in [previous, desired, (self.compact_theta, self.compact_phi)]:
            raw.extend(
                [
                    (base_theta, base_phi),
                    (base_theta - theta_step, base_phi),
                    (base_theta + theta_step, base_phi),
                    (base_theta, base_phi - phi_step),
                    (base_theta, base_phi + phi_step),
                ]
            )
        for theta in theta_values:
            for phi in phi_values:
                raw.append((theta, phi))

        out = []
        seen = set()
        for theta, phi in active_seeds + raw:
            theta = min(max(theta, theta_min), theta_max)
            phi = self._normalize_angle_deg(phi)
            if phi < phi_min:
                phi = phi_min
            if phi > phi_max:
                phi = phi_max
            key = (round(theta, 3), round(phi, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append((theta, phi))

        def prior_cost(item):
            theta, phi = item
            continuity = abs(theta - previous[0]) + 0.25 * abs(self._angle_delta(previous[1], phi))
            desired_cost = abs(theta - desired[0]) + 0.25 * abs(self._angle_delta(phi, desired[1]))
            home_cost = abs(theta - self.compact_theta) + 0.25 * abs(self._angle_delta(phi, self.compact_phi))
            if morphology_activation > 0.5:
                active_theta_cost = abs(theta - max(self.morphology_min_active_theta, previous[0]))
                return (active_theta_cost + 0.25 * continuity + 0.15 * desired_cost, desired_cost, continuity)
            return (home_cost + 0.25 * continuity + 0.1 * desired_cost, continuity, desired_cost)

        if morphology_activation > 0.5:
            seed_keys = set()
            for theta, phi in active_seeds:
                theta = min(max(theta, theta_min), theta_max)
                phi = self._normalize_angle_deg(phi)
                if phi < phi_min:
                    phi = phi_min
                if phi > phi_max:
                    phi = phi_max
                key = (round(theta, 3), round(phi, 3))
                seed_keys.add(key)
            seeds = [item for item in out if (round(item[0], 3), round(item[1], 3)) in seed_keys]
            rest = [item for item in out if (round(item[0], 3), round(item[1], 3)) not in seed_keys]
            rest.sort(key=prior_cost)
            return seeds + rest

        out.sort(key=prior_cost)
        return out

    def _desired_transit_shape(self, idx, count, goal_theta, goal_phi):
        if idx == 0:
            return (self.current_theta_phi[0], self.current_theta_phi[1])
        # Transit should reach the standoff pose in a safe flight morphology.
        # The dedicated reach-in stage below then changes theta/phi at the
        # fixed standoff pose until the tip reaches the task target.
        return (self.compact_theta, self.compact_phi)

    def _arm_morphology_metrics(self, pose, theta, phi):
        base = self._pose_xyz(pose)
        pts = world_arm_points(pose, theta, phi, self.arm_params)
        if not pts:
            return 0.0, 0.0
        cx = sum(p[0] for p in pts) / float(len(pts))
        cy = sum(p[1] for p in pts) / float(len(pts))
        com_offset = math.sqrt((cx - base[0]) * (cx - base[0]) + (cy - base[1]) * (cy - base[1]))
        lateral_extent = max(math.sqrt((p[0] - base[0]) * (p[0] - base[0]) + (p[1] - base[1]) * (p[1] - base[1])) for p in pts)
        return com_offset, lateral_extent

    def _average_points(self, points):
        if not points:
            return (0.0, 0.0, 0.0)
        inv = 1.0 / float(len(points))
        return (
            sum(p[0] for p in points) * inv,
            sum(p[1] for p in points) * inv,
            sum(p[2] for p in points) * inv,
        )

    def _local_arm_com_points(self, theta, phi):
        theta_rad = math.radians(theta)
        phi_rad = math.radians(phi)
        base_z = float(self.arm_params.get("base_offset_z", 0.10))
        length = float(self.arm_params.get("arc_length", 0.22))
        gripper = float(self.arm_params.get("gripper_length", 0.16))
        segments = max(8, int(self.arm_params.get("segment_count", 12)))
        base_com = (0.0, 0.0, 0.5 * base_z)

        flex_pts = arm_points(theta_rad, phi_rad, length, segments, base_z, 0.0)
        flex_only = [p for p in flex_pts if p[2] >= base_z - 1e-6]
        flex_com = self._average_points(flex_only)

        flex_tip = flex_pts[-1] if flex_pts else (0.0, 0.0, base_z + length)
        if abs(theta_rad) < 1e-6:
            tangent = (0.0, 0.0, 1.0)
        else:
            tangent = (
                math.sin(theta_rad) * math.cos(phi_rad),
                math.sin(theta_rad) * math.sin(phi_rad),
                math.cos(theta_rad),
            )
        tool_com = (
            flex_tip[0] + 0.5 * gripper * tangent[0],
            flex_tip[1] + 0.5 * gripper * tangent[1],
            flex_tip[2] + 0.5 * gripper * tangent[2],
        )
        return base_com, flex_com, tool_com

    def _coupling_metrics(self, pose, theta, phi):
        cache_key = (
            round(float(pose.position.x), 3),
            round(float(pose.position.y), 3),
            round(float(pose.position.z), 3),
            round(float(self._pose_yaw(pose)), 3),
            round(float(theta), 2),
            round(float(phi), 2),
        )
        cached = self._coupling_cache.get(cache_key)
        if cached is not None:
            return cached
        body = pose_to_matrix(pose)
        mount = mount_matrix(self.arm_params["mount_xyz"], self.arm_params["mount_rpy"])
        base_mat = body.dot(mount)
        base_com_local, flex_com_local, tool_com_local = self._local_arm_com_points(theta, phi)
        uav_com = self._pose_xyz(pose)
        base_com = transform_point(base_mat, base_com_local)
        flex_com = transform_point(base_mat, flex_com_local)
        tool_com = transform_point(base_mat, tool_com_local)
        total = max(self.m_uav + self.m_arm_base + self.m_flexible_section + self.m_tool, 1e-6)
        system_com = (
            (self.m_uav * uav_com[0] + self.m_arm_base * base_com[0] + self.m_flexible_section * flex_com[0] + self.m_tool * tool_com[0]) / total,
            (self.m_uav * uav_com[1] + self.m_arm_base * base_com[1] + self.m_flexible_section * flex_com[1] + self.m_tool * tool_com[1]) / total,
            (self.m_uav * uav_com[2] + self.m_arm_base * base_com[2] + self.m_flexible_section * flex_com[2] + self.m_tool * tool_com[2]) / total,
        )
        dx = system_com[0] - uav_com[0]
        dy = system_com[1] - uav_com[1]
        offset = math.sqrt(dx * dx + dy * dy)
        torque = total * 9.81 * offset
        result = {
            "system_com": system_com,
            "com_offset": offset,
            "gravity_torque": torque,
            "base_com": base_com,
            "flex_com": flex_com,
            "tool_com": tool_com,
        }
        if len(self._coupling_cache) < 300000:
            self._coupling_cache[cache_key] = result
        return result

    def _segment_dt(self, a, b, scale=1.0):
        return self._segment_dt_for_motion(a, b, None, None, scale)

    def _segment_dt_for_motion(self, a, b, shape_a=None, shape_b=None, scale=1.0):
        dist = point_distance(self._pose_xyz(a), self._pose_xyz(b))
        dt = max(self.segment_dt_min, dist / max(self.nominal_speed, 1e-3))
        yaw_delta = abs(math.atan2(math.sin(self._pose_yaw(b) - self._pose_yaw(a)), math.cos(self._pose_yaw(b) - self._pose_yaw(a))))
        dt = max(dt, yaw_delta / max(self.yaw_rate_max, 1e-3))
        if shape_a is not None and shape_b is not None:
            dtheta = abs(shape_b[0] - shape_a[0])
            dphi = abs(self._angle_delta(shape_a[1], shape_b[1]))
            dt = max(dt, dtheta / max(self.theta_dot_max, 1e-3))
            dt = max(dt, dphi / max(self.phi_dot_max, 1e-3))
        return min(self.segment_dt_max * scale, max(self.segment_dt_min, dt * scale))

    def _trajectory_dynamics_report(self, poses, arm_shapes, time_scale=1.0):
        report = {
            "dynamics_feasible": True,
            "arm_motion_feasible": True,
            "time_scale": time_scale,
            "max_velocity": 0.0,
            "max_acceleration": 0.0,
            "max_jerk": 0.0,
            "max_thrust": self.m_system_total * 9.81,
            "min_thrust": self.m_system_total * 9.81,
            "max_tilt": 0.0,
            "max_yaw_rate": 0.0,
            "max_yaw_acc": 0.0,
            "max_theta_dot": 0.0,
            "max_phi_dot": 0.0,
            "max_theta_acc": 0.0,
            "max_phi_acc": 0.0,
            "failure_reason": "none",
            "failure_segment": -1,
        }
        if len(poses) < 2:
            return report
        velocities = []
        accelerations = []
        yaw_rates = []
        theta_rates = []
        phi_rates = []
        dts = []
        for idx in range(1, len(poses)):
            dt = self._segment_dt_for_motion(poses[idx - 1], poses[idx], arm_shapes[idx - 1], arm_shapes[idx], time_scale)
            dts.append(dt)
            ax, ay, az = self._pose_xyz(poses[idx - 1])
            bx, by, bz = self._pose_xyz(poses[idx])
            vx = (bx - ax) / dt
            vy = (by - ay) / dt
            vz = (bz - az) / dt
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            report["max_velocity"] = max(report["max_velocity"], speed)
            velocities.append((vx, vy, vz))
            yaw_delta = math.atan2(math.sin(self._pose_yaw(poses[idx]) - self._pose_yaw(poses[idx - 1])), math.cos(self._pose_yaw(poses[idx]) - self._pose_yaw(poses[idx - 1])))
            yaw_rate = abs(yaw_delta) / dt
            yaw_rates.append(yaw_rate)
            report["max_yaw_rate"] = max(report["max_yaw_rate"], yaw_rate)
            theta_rate = abs(arm_shapes[idx][0] - arm_shapes[idx - 1][0]) / dt
            phi_rate = abs(self._angle_delta(arm_shapes[idx - 1][1], arm_shapes[idx][1])) / dt
            theta_rates.append(theta_rate)
            phi_rates.append(phi_rate)
            report["max_theta_dot"] = max(report["max_theta_dot"], theta_rate)
            report["max_phi_dot"] = max(report["max_phi_dot"], phi_rate)
        for idx in range(1, len(velocities)):
            dt = max(1e-3, 0.5 * (dts[idx - 1] + dts[idx]))
            av = (
                (velocities[idx][0] - velocities[idx - 1][0]) / dt,
                (velocities[idx][1] - velocities[idx - 1][1]) / dt,
                (velocities[idx][2] - velocities[idx - 1][2]) / dt,
            )
            acc = math.sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2])
            accelerations.append(av)
            report["max_acceleration"] = max(report["max_acceleration"], acc)
            thrust_vec = (av[0], av[1], av[2] + 9.81)
            thrust_norm = math.sqrt(thrust_vec[0] ** 2 + thrust_vec[1] ** 2 + thrust_vec[2] ** 2)
            thrust = self.m_system_total * thrust_norm
            tilt = math.atan2(math.sqrt(thrust_vec[0] ** 2 + thrust_vec[1] ** 2), max(thrust_vec[2], 1e-6))
            report["max_thrust"] = max(report["max_thrust"], thrust)
            report["min_thrust"] = min(report["min_thrust"], thrust)
            report["max_tilt"] = max(report["max_tilt"], tilt)
            yaw_acc = abs(yaw_rates[idx] - yaw_rates[idx - 1]) / dt
            report["max_yaw_acc"] = max(report["max_yaw_acc"], yaw_acc)
            theta_acc = abs(theta_rates[idx] - theta_rates[idx - 1]) / dt
            phi_acc = abs(phi_rates[idx] - phi_rates[idx - 1]) / dt
            report["max_theta_acc"] = max(report["max_theta_acc"], theta_acc)
            report["max_phi_acc"] = max(report["max_phi_acc"], phi_acc)
        for idx in range(1, len(accelerations)):
            dt = max(1e-3, dts[min(idx, len(dts) - 1)])
            jerk_vec = (
                (accelerations[idx][0] - accelerations[idx - 1][0]) / dt,
                (accelerations[idx][1] - accelerations[idx - 1][1]) / dt,
                (accelerations[idx][2] - accelerations[idx - 1][2]) / dt,
            )
            jerk = math.sqrt(jerk_vec[0] ** 2 + jerk_vec[1] ** 2 + jerk_vec[2] ** 2)
            report["max_jerk"] = max(report["max_jerk"], jerk)
        checks = [
            ("velocity", report["max_velocity"], self.v_max),
            ("acceleration", report["max_acceleration"], self.a_max),
            ("jerk", report["max_jerk"], self.j_max),
            ("thrust_max", report["max_thrust"], self.thrust_max),
            ("tilt", report["max_tilt"], self.tilt_max),
            ("yaw_rate", report["max_yaw_rate"], self.yaw_rate_max),
            ("yaw_acc", report["max_yaw_acc"], self.yaw_acc_max),
        ]
        for name, value, limit in checks:
            if value > limit + 1e-9:
                report["dynamics_feasible"] = False
                report["failure_reason"] = "%s_limit" % name
                break
        if report["min_thrust"] < self.thrust_min - 1e-9:
            report["dynamics_feasible"] = False
            report["failure_reason"] = "thrust_min_limit"
        arm_checks = [
            ("theta_dot", report["max_theta_dot"], self.theta_dot_max),
            ("phi_dot", report["max_phi_dot"], self.phi_dot_max),
            ("theta_acc", report["max_theta_acc"], self.theta_acc_max),
            ("phi_acc", report["max_phi_acc"], self.phi_acc_max),
        ]
        for name, value, limit in arm_checks:
            if value > limit + 1e-9:
                report["arm_motion_feasible"] = False
                if report["failure_reason"] == "none":
                    report["failure_reason"] = "%s_limit" % name
                break
        return report

    def _refine_time_allocation(self, poses, arm_shapes):
        if not self.time_allocation_refine_enabled:
            return self._trajectory_dynamics_report(poses, arm_shapes, 1.0)
        scale = 1.0
        report = self._trajectory_dynamics_report(poses, arm_shapes, scale)
        while (not report["dynamics_feasible"] or not report["arm_motion_feasible"]) and scale < self.time_refine_max_scale - 1e-9:
            scale = min(self.time_refine_max_scale, scale * 1.25)
            report = self._trajectory_dynamics_report(poses, arm_shapes, scale)
        return report

    def _sequence_edge_swept_safe(self, prev, curr):
        return not self._sequence_edge_required_unsafe(prev, curr)

    def _sequence_edge_dense_check_needed(self, prev, curr):
        if not self.sequence_edge_collision_lazy:
            return True
        min_body = min(prev["body_margin"], curr["body_margin"])
        min_arm = min(prev["arm_margin"], curr["arm_margin"])
        body_edge_target = max(self.sequence_risk_body_margin, self.edge_body_margin_target)
        arm_edge_target = max(self.sequence_risk_arm_margin, self.edge_arm_margin_target)
        if min_body < body_edge_target + 0.015 or min_arm < arm_edge_target + 0.015:
            return True
        dist = point_distance(self._pose_xyz(prev["pose"]), self._pose_xyz(curr["pose"]))
        if dist > 0.30 and min(min_body, min_arm) < min(body_edge_target, arm_edge_target) + 0.04:
            return True
        dtheta = abs(curr["shape"][0] - prev["shape"][0])
        dphi = abs(self._angle_delta(prev["shape"][1], curr["shape"][1]))
        return (dtheta > self.swept_theta_step or dphi > self.swept_phi_step) and min(min_body, min_arm) < min(body_edge_target, arm_edge_target) + 0.03

    def _sequence_edge_required_unsafe(self, prev, curr):
        prev_pose = prev["pose"]
        curr_pose = curr["pose"]
        prev_shape = prev["shape"]
        curr_shape = curr["shape"]
        cache_key = (
            "required",
            round(float(prev_pose.position.x), 3),
            round(float(prev_pose.position.y), 3),
            round(float(prev_pose.position.z), 3),
            round(float(self._pose_yaw(prev_pose)), 3),
            round(float(prev_shape[0]), 2),
            round(float(prev_shape[1]), 2),
            round(float(curr_pose.position.x), 3),
            round(float(curr_pose.position.y), 3),
            round(float(curr_pose.position.z), 3),
            round(float(self._pose_yaw(curr_pose)), 3),
            round(float(curr_shape[0]), 2),
            round(float(curr_shape[1]), 2),
            bool(self.cloud_collision_enabled),
        )
        cached = self._edge_safety_cache.get(cache_key)
        if cached is not None:
            return cached
        dist = point_distance(self._pose_xyz(prev_pose), self._pose_xyz(curr_pose))
        dtheta = abs(curr_shape[0] - prev_shape[0])
        dphi = abs(self._angle_delta(prev_shape[1], curr_shape[1]))
        steps = max(
            2,
            int(math.ceil(dist / max(self.swept_translation_step, 1e-3))),
            int(math.ceil(dtheta / max(self.swept_theta_step, 1e-3))),
            int(math.ceil(dphi / max(self.swept_phi_step, 1e-3))),
        )
        unsafe = False
        for step in range(1, steps):
            ratio = float(step) / float(steps)
            pose = self._interpolate_pose_between(prev_pose, curr_pose, ratio)
            theta = prev_shape[0] + (curr_shape[0] - prev_shape[0]) * ratio
            phi = self._normalize_angle_deg(prev_shape[1] + self._angle_delta(prev_shape[1], curr_shape[1]) * ratio)
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            if body_margin < self.required_body_margin:
                unsafe = True
                break
            arm_margin, _, _ = self._arm_margin_detail(pose, theta, phi, sample_limit=7)
            if arm_margin < self.required_arm_margin:
                unsafe = True
                break
        if len(self._edge_safety_cache) < 300000:
            self._edge_safety_cache[cache_key] = unsafe
        return unsafe

    def _sequence_edge_required_detail(self, prev, curr):
        prev_pose = prev["pose"]
        curr_pose = curr["pose"]
        prev_shape = prev["shape"]
        curr_shape = curr["shape"]
        dist = point_distance(self._pose_xyz(prev_pose), self._pose_xyz(curr_pose))
        dtheta = abs(curr_shape[0] - prev_shape[0])
        dphi = abs(self._angle_delta(prev_shape[1], curr_shape[1]))
        steps = max(
            2,
            int(math.ceil(dist / max(self.swept_translation_step, 1e-3))),
            int(math.ceil(dtheta / max(self.swept_theta_step, 1e-3))),
            int(math.ceil(dphi / max(self.swept_phi_step, 1e-3))),
        )
        detail = {
            "dist": dist,
            "dtheta": dtheta,
            "dphi": dphi,
            "steps": steps,
            "min_body_margin": float("inf"),
            "min_body_obstacle": "none",
            "min_body_point": (0.0, 0.0, 0.0),
            "min_arm_margin": float("inf"),
            "min_arm_obstacle": "none",
            "min_arm_point": (0.0, 0.0, 0.0),
        }
        for step in range(steps + 1):
            ratio = float(step) / float(max(steps, 1))
            pose = self._interpolate_pose_between(prev_pose, curr_pose, ratio)
            theta = prev_shape[0] + (curr_shape[0] - prev_shape[0]) * ratio
            phi = self._normalize_angle_deg(prev_shape[1] + self._angle_delta(prev_shape[1], curr_shape[1]) * ratio)
            body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
            if body_margin < detail["min_body_margin"]:
                detail["min_body_margin"] = body_margin
                detail["min_body_obstacle"] = body_obstacle
                detail["min_body_point"] = body_point
            arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(pose, theta, phi, sample_limit=7)
            if arm_margin < detail["min_arm_margin"]:
                detail["min_arm_margin"] = arm_margin
                detail["min_arm_obstacle"] = arm_obstacle
                detail["min_arm_point"] = arm_point
        return detail

    def _annotate_transition_failure(self, failure, prev_layer, curr_candidates):
        failure["transition_prev_candidate_count"] = len(prev_layer)
        failure["transition_curr_candidate_count"] = len(curr_candidates)
        best = None
        best_score = -float("inf")
        for _, prev_state in prev_layer[: min(len(prev_layer), 12)]:
            prev_cand = prev_state["candidate"]
            for curr_cand in curr_candidates[: min(len(curr_candidates), 8)]:
                detail = self._sequence_edge_required_detail(prev_cand, curr_cand)
                score = min(detail["min_body_margin"], detail["min_arm_margin"])
                if score > best_score:
                    best_score = score
                    best = (prev_cand, curr_cand, detail)
        if best is None:
            return
        prev_cand, curr_cand, detail = best
        failure["transition_prev_pose"] = self._pose_xyz(prev_cand["pose"])
        failure["transition_curr_pose"] = self._pose_xyz(curr_cand["pose"])
        failure["transition_prev_shape"] = prev_cand["shape"]
        failure["transition_curr_shape"] = curr_cand["shape"]
        failure["transition_edge_dist"] = detail["dist"]
        failure["transition_edge_dtheta"] = detail["dtheta"]
        failure["transition_edge_dphi"] = detail["dphi"]
        failure["transition_edge_steps"] = detail["steps"]
        failure["transition_min_body_margin"] = detail["min_body_margin"]
        failure["transition_min_body_obstacle"] = detail["min_body_obstacle"]
        failure["transition_min_body_point"] = detail["min_body_point"]
        failure["transition_min_arm_margin"] = detail["min_arm_margin"]
        failure["transition_min_arm_obstacle"] = detail["min_arm_obstacle"]
        failure["transition_min_arm_point"] = detail["min_arm_point"]

    def _sequence_edge_safety_penalty(self, prev, curr):
        prev_pose = prev["pose"]
        curr_pose = curr["pose"]
        prev_shape = prev["shape"]
        curr_shape = curr["shape"]
        cache_key = (
            "penalty",
            round(float(prev_pose.position.x), 3),
            round(float(prev_pose.position.y), 3),
            round(float(prev_pose.position.z), 3),
            round(float(self._pose_yaw(prev_pose)), 3),
            round(float(prev_shape[0]), 2),
            round(float(prev_shape[1]), 2),
            round(float(curr_pose.position.x), 3),
            round(float(curr_pose.position.y), 3),
            round(float(curr_pose.position.z), 3),
            round(float(self._pose_yaw(curr_pose)), 3),
            round(float(curr_shape[0]), 2),
            round(float(curr_shape[1]), 2),
            bool(self.cloud_collision_enabled),
        )
        cached = self._edge_safety_cache.get(cache_key)
        if cached is not None:
            return cached
        penalty = 0.0
        body_target = max(self.required_body_margin, self.edge_body_margin_target)
        arm_target = max(self.required_arm_margin, self.edge_arm_margin_target)
        dist = point_distance(self._pose_xyz(prev_pose), self._pose_xyz(curr_pose))
        dtheta = abs(curr_shape[0] - prev_shape[0])
        dphi = abs(self._angle_delta(prev_shape[1], curr_shape[1]))
        steps = max(
            4,
            int(math.ceil(dist / max(self.swept_translation_step, 1e-3))),
            int(math.ceil(dtheta / max(self.swept_theta_step, 1e-3))),
            int(math.ceil(dphi / max(self.swept_phi_step, 1e-3))),
        )
        for step in range(1, steps):
            ratio = float(step) / float(steps)
            pose = self._interpolate_pose_between(prev_pose, curr_pose, ratio)
            theta = prev_shape[0] + (curr_shape[0] - prev_shape[0]) * ratio
            phi = self._normalize_angle_deg(prev_shape[1] + self._angle_delta(prev_shape[1], curr_shape[1]) * ratio)
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            if body_margin < body_target:
                deficit = (body_target - body_margin) / max(body_target, 1e-3)
                penalty += 220.0 * deficit * deficit
            arm_margin, _, _ = self._arm_margin_detail(pose, theta, phi, sample_limit=5)
            if arm_margin < arm_target:
                deficit = (arm_target - arm_margin) / max(arm_target, 1e-3)
                penalty += 260.0 * deficit * deficit
        if len(self._edge_safety_cache) < 300000:
            self._edge_safety_cache[cache_key] = penalty
        return penalty

    def _swept_validation(self, poses, arm_shapes):
        result = {
            "swept_validation": bool(self.swept_validation_enabled),
            "swept_feasible": True,
            "min_swept_body_margin": float("inf"),
            "min_swept_body_obstacle": "none",
            "min_swept_body_point": (0.0, 0.0, 0.0),
            "min_swept_arm_margin": float("inf"),
            "min_swept_arm_obstacle": "none",
            "min_swept_arm_point": (0.0, 0.0, 0.0),
            "swept_failure_segment": -1,
            "swept_sample_count": 0,
        }
        if not self.swept_validation_enabled or len(poses) < 2:
            return result
        for idx in range(1, len(poses)):
            dist = point_distance(self._pose_xyz(poses[idx - 1]), self._pose_xyz(poses[idx]))
            dtheta = abs(arm_shapes[idx][0] - arm_shapes[idx - 1][0])
            dphi = abs(self._angle_delta(arm_shapes[idx - 1][1], arm_shapes[idx][1]))
            steps = max(
                1,
                int(math.ceil(dist / self.swept_translation_step)),
                int(math.ceil(dtheta / self.swept_theta_step)),
                int(math.ceil(dphi / self.swept_phi_step)),
            )
            for step in range(steps + 1):
                ratio = float(step) / float(max(steps, 1))
                pose = self._interpolate_pose_between(poses[idx - 1], poses[idx], ratio)
                theta = arm_shapes[idx - 1][0] + (arm_shapes[idx][0] - arm_shapes[idx - 1][0]) * ratio
                phi = self._normalize_angle_deg(arm_shapes[idx - 1][1] + self._angle_delta(arm_shapes[idx - 1][1], arm_shapes[idx][1]) * ratio)
                body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
                arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(pose, theta, phi)
                result["swept_sample_count"] += 1
                if body_margin < result["min_swept_body_margin"]:
                    result["min_swept_body_margin"] = body_margin
                    result["min_swept_body_obstacle"] = body_obstacle
                    result["min_swept_body_point"] = body_point
                if arm_margin < result["min_swept_arm_margin"]:
                    result["min_swept_arm_margin"] = arm_margin
                    result["min_swept_arm_obstacle"] = arm_obstacle
                    result["min_swept_arm_point"] = arm_point
                if body_margin < self.required_body_margin or arm_margin < self.required_arm_margin:
                    result["swept_feasible"] = False
                    if result["swept_failure_segment"] < 0:
                        result["swept_failure_segment"] = idx - 1
        return result

    def _local_refinement_window(self, poses):
        if len(poses) <= self.refine_max_points and not self.sequence_optimizer_enabled:
            return poses
        if self.sequence_optimizer_enabled:
            return self._keyframe_refinement_window(poses)
        out = []
        last_idx = -1
        for sample_idx in range(self.refine_max_points):
            idx = int(round(sample_idx * (len(poses) - 1) / float(max(self.refine_max_points - 1, 1))))
            if idx == last_idx:
                continue
            out.append(poses[idx])
            last_idx = idx
        return out

    def _keyframe_refinement_window(self, poses):
        max_points = min(self.refine_max_points, self.sequence_refine_max_points, 24)
        if len(poses) <= max_points:
            return poses
        keep = {0, len(poses) - 1}
        for slot in self._slot_constraints():
            guard = slot["half_y"] + max(0.12, self.body_radius + self.body_clearance)
            for target_y in (slot["y"] - guard, slot["y"], slot["y"] + guard):
                best_idx = None
                best_dist = float("inf")
                for idx, pose in enumerate(poses):
                    x, y, _ = self._pose_xyz(pose)
                    if x < slot["x_min"] - 0.60 or x > slot["x_max"] + 0.60:
                        continue
                    dist = abs(y - target_y)
                    if dist < best_dist:
                        best_idx = idx
                        best_dist = dist
                if best_idx is not None:
                    keep.add(best_idx)
        scores = []
        for idx, pose in enumerate(poses):
            body_margin, _, _ = self._body_point_margin(self._pose_xyz(pose))
            arm_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=5)
            clearance_score = max(0.0, self.refine_clearance_target - min(body_margin, arm_margin))
            robust_score = max(0.0, self.required_body_margin - body_margin) + max(0.0, self.required_arm_margin - arm_margin)
            curvature_score = 0.0
            if 0 < idx < len(poses) - 1:
                a = self._pose_xyz(poses[idx - 1])
                b = self._pose_xyz(pose)
                c = self._pose_xyz(poses[idx + 1])
                yaw1 = math.atan2(b[1] - a[1], b[0] - a[0])
                yaw2 = math.atan2(c[1] - b[1], c[0] - b[0])
                curvature_score = abs(math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1)))
            z_change_score = 0.0
            if 0 < idx < len(poses) - 1:
                z_change_score = abs(poses[idx + 1].position.z - poses[idx - 1].position.z)
            score = 8.0 * robust_score + 4.0 * clearance_score + curvature_score + z_change_score
            if score > 1e-6:
                scores.append((score, idx))

        scores.sort(reverse=True)
        for _, idx in scores:
            keep.add(idx)
            if idx > 0:
                keep.add(idx - 1)
            if idx < len(poses) - 1:
                keep.add(idx + 1)
            if len(keep) >= max_points:
                break
        if len(keep) < max_points:
            for sample_idx in range(max_points):
                idx = int(round(sample_idx * (len(poses) - 1) / float(max(max_points - 1, 1))))
                keep.add(idx)
                if len(keep) >= max_points:
                    break
        return [self._copy_pose(poses[idx]) for idx in sorted(keep)]

    def _ordered_pose_offsets(self, lateral_offsets, z_offsets):
        offsets = [(lat, dz) for lat in lateral_offsets for dz in z_offsets]
        # In low-clearance passages the arm usually fails against lower
        # obstacles while the UAV body still has ceiling margin. Try small
        # upward pose repairs before spending the waypoint budget on lateral
        # detours or downward offsets.
        return sorted(
            offsets,
            key=lambda item: (
                abs(item[0]) + 0.7 * max(item[1], 0.0) + 4.0 * max(-item[1], 0.0),
                abs(item[0]),
                -item[1],
            ),
        )

    def _ordered_yaw_offsets(self, yaw_offsets):
        return sorted(yaw_offsets, key=lambda value: abs(value))

    def _new_refinement_info(self, ego_poses):
        return {
            "whole_body_refinement_used": False,
            "refinement_input_points": len(ego_poses),
            "pose_refined_count": 0,
            "arm_optimized_count": 0,
            "max_ego_deviation": 0.0,
            "mean_ego_deviation": 0.0,
            "max_arm_com_offset": 0.0,
            "mean_arm_com_offset": 0.0,
            "max_arm_lateral_extent": 0.0,
            "max_gravity_torque": 0.0,
            "mean_gravity_torque": 0.0,
            "transit_compact_weight": self.refine_compact_weight,
            "transit_com_weight": self.refine_com_weight,
            "transit_clearance_weight": self.refine_clearance_weight,
            "refine_min_arm_margin": self.refine_min_arm_margin,
            "required_body_margin": self.required_body_margin,
            "required_arm_margin": self.required_arm_margin,
            "hard_com_limit": self.hard_com_limit,
            "tau_limit": self.tau_limit,
            "morphology_gate": "home_clearance_activated",
            "morphology_home_hold_weight": self.morphology_home_hold_weight,
            "morphology_activation_margin": self.morphology_activation_margin,
            "morphology_activation_band": self.morphology_activation_band,
            "morphology_min_active_theta": self.morphology_min_active_theta,
            "passage_morphology_gate_enabled": self.passage_morphology_gate_enabled,
            "passage_compact_lock_wp_count": 0,
            "passage_risk_wp_count": 0,
            "passage_compact_safe_wp_count": 0,
            "passage_segment_count": 0,
            "passage_hold_wp_count": 0,
            "passage_hold_shape_count": 0,
            "passage_hold_min_arm_margin": 0.0,
            "selected_passage_compact_lock_count": 0,
            "selected_passage_risk_count": 0,
            "release_min": float("inf"),
            "release_max": 0.0,
            "release_mean": 0.0,
            "release_at_failure": 0.0,
            "morphology_activated_wp_count": 0,
            "mean_morphology_activation": 0.0,
            "valid_candidate_target": self.refine_valid_candidate_target,
            "sequence_min_candidates_per_wp": self.sequence_min_candidates_per_wp,
            "shape_policy": "dp_graph_dynamics_aware",
            "sequence_optimizer": "dp_graph",
            "candidate_count": 0,
            "expanded_edges": 0,
            "sequence_cost": 0.0,
            "sequence_beam_width": self.sequence_beam_width,
            "sequence_effective_beam_width": self.sequence_beam_width,
            "sequence_edge_eval_neighbors": self.sequence_edge_eval_neighbors,
            "sequence_effective_transition_neighbors": self.sequence_transition_neighbors,
            "sequence_edge_collision_lazy": self.sequence_edge_collision_lazy,
            "sequence_edge_safety_in_dp": False,
            "sequence_edge_exhaustive_retries": 0,
            "sequence_dp_candidates_per_wp": self.sequence_dp_candidates_per_wp,
            "sequence_candidate_budget_fraction": self.sequence_candidate_budget_fraction,
            "sequence_dp_min_time_budget": self.sequence_dp_min_time_budget,
            "sequence_candidate_generation_sec": 0.0,
            "sequence_dp_sec": 0.0,
            "sequence_dp_candidate_count": 0,
            "sequence_dp_max_candidates_per_wp": self.sequence_dp_candidates_per_wp,
            "sequence_dp_min_pose_diversity": 0,
            "sequence_dp_max_pose_diversity": 0,
            "sequence_dp_rich_risk_layers": 0,
            "sequence_approach_release_candidate_count": 0,
            "sequence_approach_bridge_candidate_count": 0,
            "sequence_dp_partial_timeout": False,
            "sequence_dp_partial_timeout_wp": -1,
            "edge_body_margin_target": self.edge_body_margin_target,
            "edge_arm_margin_target": self.edge_arm_margin_target,
            "sequence_risk_body_margin": self.sequence_risk_body_margin,
            "sequence_risk_arm_margin": self.sequence_risk_arm_margin,
            "transition_repair_inserted": 0,
            "swept_repair_inserted": 0,
            "local_repair_attempts": 0,
            "edge_repair_enabled": self.edge_repair_enabled,
            "edge_repair_used": False,
            "edge_repair_attempts": 0,
            "edge_repair_successes": 0,
            "edge_repair_inserted_knots": 0,
            "edge_repair_min_body_margin": float("inf"),
            "edge_repair_min_arm_margin": float("inf"),
            "edge_repair_time_budget": self.edge_repair_time_budget,
            "edge_repair_timeout": False,
            "terminal_bridge_attempts": 0,
            "terminal_bridge_successes": 0,
            "terminal_bridge_knots": 0,
            "terminal_bridge_best_body_margin": float("-inf"),
            "terminal_bridge_best_body_obstacle": "none",
            "terminal_bridge_best_body_point": (0.0, 0.0, 0.0),
            "terminal_bridge_best_arm_margin": float("-inf"),
            "terminal_bridge_best_arm_obstacle": "none",
            "terminal_bridge_best_arm_point": (0.0, 0.0, 0.0),
            "terminal_bridge_best_status": "none",
            "refine_first_feasible": self.refine_first_feasible,
            "refine_time_budget": self.refine_time_budget,
            "refine_wp_time_budget": self.refine_wp_time_budget,
            "shape_candidates_per_pose": self.refine_shape_candidates_per_pose,
            "timeout_sec": 0.0,
            "failure_wp": -1,
            "failure_pose": (0.0, 0.0, 0.0),
            "failure_body_best_margin": float("-inf"),
            "failure_body_obstacle": "none",
            "failure_body_point": (0.0, 0.0, 0.0),
            "failure_arm_best_margin": float("-inf"),
            "failure_arm_obstacle": "none",
            "failure_arm_point": (0.0, 0.0, 0.0),
            "failure_pose_candidates": 0,
            "failure_body_valid_candidates": 0,
            "failure_shape_candidates": 0,
            "failure_arm_valid_candidates": 0,
            "failure_reason": "none",
            "failure_passage_risk": False,
            "failure_passage_compact_lock": False,
            "failure_passage_compact_safe": False,
            "failure_passage_slot": "none",
            "failure_release_factor": 0.0,
            "failure_approach_release": False,
            "failure_stable_shape": (0.0, 0.0),
            "failure_q_ref": (0.0, 0.0),
            "failure_passage_hold_used": False,
            "failure_passage_hold_shape": (0.0, 0.0),
            "failure_passage_hold_segment_id": -1,
            "min_candidate_body_margin": float("inf"),
            "min_candidate_arm_margin": float("inf"),
        }

    def _shape_candidates_for_sequence(self, idx, count, desired_shape, morphology_activation, passage_context=None):
        previous = (self.current_theta_phi[0], self.current_theta_phi[1]) if idx == 0 else (self.compact_theta, self.compact_phi)
        candidates = self._shape_candidates(previous, desired_shape, morphology_activation)
        if passage_context:
            extra = [
                passage_context.get("stable_shape", (self.compact_theta, self.compact_phi)),
                passage_context.get("q_ref", desired_shape),
                passage_context.get("pre_reach_shape", desired_shape),
            ]
            if (
                passage_context.get("approach_release", False)
                or (
                    passage_context.get("passage_slot", "none") == "none"
                    and float(passage_context.get("release_factor", 0.0)) > 0.35
                )
            ):
                extra.extend(
                    self._release_bridge_shape_candidates(
                        passage_context.get("stable_shape", (self.compact_theta, self.compact_phi)),
                        passage_context.get("pre_reach_shape", desired_shape),
                        passage_context.get("q_ref", desired_shape),
                    )
                )
            merged = []
            seen = set()
            for theta, phi in extra + candidates:
                key = (round(float(theta), 3), round(float(self._normalize_angle_deg(phi)), 3))
                if key in seen:
                    continue
                seen.add(key)
                merged.append((float(theta), self._normalize_angle_deg(float(phi))))
            candidates = merged
        return candidates[: self.refine_shape_candidates_per_pose]

    def _focused_stable_shape_candidates(self, passage_context):
        stable = passage_context.get("stable_shape", (self.compact_theta, self.compact_phi))
        q_ref = passage_context.get("q_ref", stable)
        compact = (self.compact_theta, self.compact_phi)
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)
        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        low_release_passage = (
            passage_context.get("passage_risk", False)
            and float(passage_context.get("release_factor", 0.0)) < 0.15
        )
        seeds = [stable, q_ref, self._blend_shape(stable, q_ref, 0.5), compact]
        if low_release_passage:
            # In narrow passages the useful freedom is mostly theta and body
            # pose. Wide phi sweeps inflate DP and also create harder swept
            # edges, so keep phi locked to the local stable/q_ref directions.
            low_theta = min(max(theta_min, 0.0), theta_max)
            active_low_theta = min(max(self.morphology_min_active_theta, theta_min), theta_max)
            for target_phi in [stable[1], q_ref[1]]:
                seeds.extend([(low_theta, target_phi), (active_low_theta, target_phi)])
            for base_theta, base_phi in [stable, q_ref]:
                seeds.extend([(base_theta - theta_step, base_phi), (base_theta + theta_step, base_phi)])
        elif passage_context.get("approach_release", False):
            reach = passage_context.get("pre_reach_shape", q_ref)
            seeds.extend(self._release_bridge_shape_candidates(stable, reach, q_ref))
        else:
            for base_theta, base_phi in [stable, q_ref]:
                seeds.extend(
                    [
                        (base_theta - theta_step, base_phi),
                        (base_theta + theta_step, base_phi),
                        (base_theta, base_phi - phi_step),
                        (base_theta, base_phi + phi_step),
                    ]
                )
        out = []
        seen = set()
        for theta, phi in seeds:
            theta = min(max(float(theta), theta_min), theta_max)
            phi = self._normalize_angle_deg(float(phi))
            key = (round(theta, 3), round(phi, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append((theta, phi))
            if low_release_passage and len(out) >= 6:
                break
        return out

    def _slot_shape_seeds(self, pose):
        slot = self._slot_constraint_for_pose(pose)
        if slot is None:
            return []
        theta_max = float(self.arm_params["theta_max_deg"])
        theta_low = max(self.morphology_min_active_theta, min(theta_max, 45.0))
        theta_mid = max(self.morphology_min_active_theta, min(theta_max, 75.0))
        theta_high = min(theta_max, max(theta_mid, 105.0))
        theta_max_seed = min(theta_max, 120.0)
        seeds = []
        for theta in (theta_max_seed, theta_high, theta_mid, theta_low):
            for phi in (0.0, 90.0, -90.0, 180.0, -180.0, 60.0, -60.0, 120.0, -120.0):
                seeds.append((theta, self._normalize_angle_deg(phi)))
        out = []
        seen = set()
        for theta, phi in seeds:
            key = (round(theta, 3), round(phi, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append((theta, phi))
        return out

    def _try_add_sequence_candidate(
        self,
        candidates,
        failure,
        pose,
        shape,
        ego_pose,
        desired_shape,
        body_margin,
        body_obstacle,
        body_point,
        morphology_activation,
        passage_context=None,
    ):
        theta, phi = shape
        failure["shape_candidates"] += 1
        arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(pose, theta, phi)
        if arm_margin > failure["arm_best_margin"]:
            failure["arm_best_margin"] = arm_margin
            failure["arm_obstacle"] = arm_obstacle
            failure["arm_point"] = arm_point
        if arm_margin < self.required_arm_margin:
            failure["failure_reason"] = "arm_margin"
            return False
        candidate = self._make_sequence_candidate(
            pose,
            (theta, phi),
            ego_pose,
            desired_shape,
            body_margin,
            body_obstacle,
            body_point,
            arm_margin,
            arm_obstacle,
            arm_point,
            morphology_activation,
            passage_context,
        )
        if candidate is None:
            failure["failure_reason"] = "coupling_limit"
            return False
        candidates.append(candidate)
        failure["arm_valid_candidates"] += 1
        return True

    def _candidate_node_cost(self, candidate, desired_shape, ego_pose, morphology_activation):
        theta, phi = candidate["shape"]
        body_margin = candidate["body_margin"]
        arm_margin = candidate["arm_margin"]
        coupling = candidate["coupling"]
        lateral_extent = float(candidate.get("lateral_extent", 0.0))
        ego_dev = point_distance(self._pose_xyz(ego_pose), self._pose_xyz(candidate["pose"]))
        desired_cost = abs(theta - desired_shape[0]) + 0.25 * abs(self._angle_delta(phi, desired_shape[1]))
        compact_cost = abs(theta - self.compact_theta) + 0.25 * abs(self._angle_delta(phi, self.compact_phi))
        q_ref = candidate.get("q_ref", desired_shape)
        stable_shape = candidate.get("stable_shape", (self.compact_theta, self.compact_phi))
        release_factor = min(1.0, max(0.0, float(candidate.get("release_factor", 0.0))))
        q_ref_cost = self._shape_distance((theta, phi), q_ref)
        stable_cost = self._shape_distance((theta, phi), stable_shape)
        bending_cost = (theta / max(float(self.arm_params["theta_max_deg"]), 1.0)) ** 2
        clearance_deficit = max(0.0, self.refine_clearance_target - min(body_margin, arm_margin)) / max(self.refine_clearance_target, 1e-3)
        robust_deficit = (
            max(0.0, self.required_body_margin - body_margin) / max(self.required_body_margin, 1e-3)
            + max(0.0, self.required_arm_margin - arm_margin) / max(self.required_arm_margin, 1e-3)
        )
        body_edge_deficit = max(0.0, self.edge_body_margin_target - body_margin) / max(self.edge_body_margin_target, 1e-3)
        arm_edge_deficit = max(0.0, self.edge_arm_margin_target - arm_margin) / max(self.edge_arm_margin_target, 1e-3)
        slot_cost = 0.0
        slot = self._slot_constraint_for_pose(candidate["pose"])
        if slot is not None:
            slot_body_target = max(self.refine_clearance_target, self.required_body_margin + 0.04)
            slot_arm_target = max(self.refine_clearance_target, self.required_arm_margin + 0.04)
            slot_body_deficit = max(0.0, slot_body_target - body_margin) / max(slot_body_target, 1e-3)
            slot_arm_deficit = max(0.0, slot_arm_target - arm_margin) / max(slot_arm_target, 1e-3)
            z_dev = abs(candidate["pose"].position.z - slot["z_target"]) / max(self.refine_z_step, 1e-3)
            slot_cost = 18.0 * slot_body_deficit * slot_body_deficit + 24.0 * slot_arm_deficit * slot_arm_deficit + 0.18 * z_dev * z_dev
        passage_bias = 1.5 if candidate.get("passage_risk", False) else 1.0
        stable_shape_cost = passage_bias * (1.0 - release_factor) * self.morphology_stability_shape_weight * stable_cost
        q_ref_shape_cost = (0.35 + release_factor) * self.morphology_q_ref_weight * q_ref_cost
        lateral_extent_cost = passage_bias * self.morphology_lateral_extent_weight * lateral_extent * lateral_extent
        com_deficit = max(0.0, coupling["com_offset"] - self.preferred_com_offset) / max(self.hard_com_limit, 1e-3)
        torque_deficit = max(0.0, coupling["gravity_torque"] - 0.7 * self.tau_limit) / max(self.tau_limit, 1e-3)
        home_hold_cost = (1.0 - morphology_activation) * self.morphology_home_hold_weight * compact_cost
        return (
            4.0 * ego_dev
            + 0.01 * desired_cost
            + self.refine_compact_weight * compact_cost
            + self.bending_weight * bending_cost
            + self.com_weight * coupling["com_offset"]
            + self.torque_weight * coupling["gravity_torque"]
            + self.refine_clearance_weight * clearance_deficit * clearance_deficit
            + 4.0 * self.refine_clearance_weight * robust_deficit * robust_deficit
            + 10.0 * body_edge_deficit * body_edge_deficit
            + 12.0 * arm_edge_deficit * arm_edge_deficit
            + slot_cost
            + 8.0 * com_deficit * com_deficit
            + 8.0 * torque_deficit * torque_deficit
            + home_hold_cost
            + stable_shape_cost
            + q_ref_shape_cost
            + lateral_extent_cost
        )

    def _candidate_pose_key(self, candidate):
        pose = candidate["pose"]
        x, y, z = self._pose_xyz(pose)
        return (
            round(float(x), 2),
            round(float(y), 2),
            round(float(z), 2),
            round(float(self._pose_yaw(pose)), 2),
        )

    def _candidate_pose_diversity(self, candidates):
        return len({self._candidate_pose_key(candidate) for candidate in candidates})

    def _approach_release_phase_key(self, candidate):
        theta, phi = candidate["shape"]
        stable = candidate.get("stable_shape", (self.compact_theta, self.compact_phi))
        q_ref = candidate.get("q_ref", stable)
        reach = candidate.get("pre_reach_shape", q_ref)
        theta_min = float(self.arm_params["theta_min_deg"])
        low_theta = min(max(theta_min, self.compact_theta), float(self.arm_params["theta_max_deg"]))
        active_low_theta = min(
            max(self.morphology_min_active_theta, theta_min),
            float(self.arm_params["theta_max_deg"]),
        )
        if abs(theta - low_theta) <= 0.5 * max(float(self.arm_params.get("theta_step_deg", 10.0)), 1.0):
            theta_key = "low"
        elif abs(theta - active_low_theta) <= 0.5 * max(float(self.arm_params.get("theta_step_deg", 10.0)), 1.0):
            theta_key = "active_low"
        elif abs(theta - reach[0]) <= 0.5 * max(float(self.arm_params.get("theta_step_deg", 10.0)), 1.0):
            theta_key = "reach_theta"
        elif abs(theta - stable[0]) <= 0.5 * max(float(self.arm_params.get("theta_step_deg", 10.0)), 1.0):
            theta_key = "stable_theta"
        else:
            theta_key = "theta_%d" % int(round(theta / 20.0))

        if abs(self._angle_delta(phi, stable[1])) <= 18.0:
            phi_key = "stable_phi"
        elif abs(self._angle_delta(phi, q_ref[1])) <= 18.0:
            phi_key = "qref_phi"
        elif abs(self._angle_delta(phi, reach[1])) <= 18.0:
            phi_key = "reach_phi"
        else:
            phi_key = "phi_%d" % int(round(self._normalize_angle_deg(phi) / 45.0))
        return (theta_key, phi_key)

    def _is_approach_bridge_candidate(self, candidate):
        if not candidate.get("approach_release", False):
            return False
        theta, phi = candidate["shape"]
        reach = candidate.get("pre_reach_shape", candidate.get("q_ref", candidate["shape"]))
        q_ref = candidate.get("q_ref", reach)
        low_like = theta <= max(self.morphology_min_active_theta + 5.0, self.compact_theta + 5.0)
        target_phi_like = (
            abs(self._angle_delta(phi, reach[1])) <= 35.0
            or abs(self._angle_delta(phi, q_ref[1])) <= 35.0
        )
        return bool(low_like or target_phi_like)

    def _select_candidate_subset(self, candidates, limit, prefer_pose_diversity=False):
        ordered = sorted(candidates, key=lambda item: item["node_cost"])
        if any(candidate.get("approach_release", False) for candidate in ordered) and len(ordered) > limit:
            selected = []
            selected_ids = set()

            def add(candidate):
                if id(candidate) in selected_ids:
                    return False
                selected.append(candidate)
                selected_ids.add(id(candidate))
                return len(selected) >= limit

            add(ordered[0])
            seen_phase_keys = set()
            for candidate in ordered:
                if not candidate.get("approach_release", False):
                    continue
                key = self._approach_release_phase_key(candidate)
                if key in seen_phase_keys:
                    continue
                seen_phase_keys.add(key)
                if add(candidate):
                    return selected
                if len(seen_phase_keys) >= max(6, min(limit - 1, 12)):
                    break
            for candidate in ordered:
                if add(candidate):
                    return selected
            return selected
        if not prefer_pose_diversity or len(ordered) <= limit:
            return ordered[:limit]
        selected = []
        selected_ids = set()
        seen_pose_keys = set()
        for candidate in ordered:
            key = self._candidate_pose_key(candidate)
            if key in seen_pose_keys:
                continue
            seen_pose_keys.add(key)
            selected.append(candidate)
            selected_ids.add(id(candidate))
            if len(selected) >= limit:
                return selected
        for candidate in ordered:
            if id(candidate) in selected_ids:
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return selected

    def _select_state_subset(self, states, limit, prefer_pose_diversity=False):
        ordered = sorted(states, key=lambda item: item["cost"])
        if any(state["candidate"].get("approach_release", False) for state in ordered) and len(ordered) > limit:
            selected = []
            selected_ids = set()

            def add(state):
                if id(state) in selected_ids:
                    return False
                selected.append(state)
                selected_ids.add(id(state))
                return len(selected) >= limit

            add(ordered[0])
            seen_phase_keys = set()
            for state in ordered:
                candidate = state["candidate"]
                if not candidate.get("approach_release", False):
                    continue
                key = self._approach_release_phase_key(candidate)
                if key in seen_phase_keys:
                    continue
                seen_phase_keys.add(key)
                if add(state):
                    return selected
                if len(seen_phase_keys) >= max(6, min(limit - 1, 12)):
                    break
            for state in ordered:
                if add(state):
                    return selected
            return selected
        if not prefer_pose_diversity or len(ordered) <= limit:
            return ordered[:limit]
        selected = []
        selected_ids = set()
        seen_pose_keys = set()
        for state in ordered:
            key = self._candidate_pose_key(state["candidate"])
            if key in seen_pose_keys:
                continue
            seen_pose_keys.add(key)
            selected.append(state)
            selected_ids.add(id(state))
            if len(selected) >= limit:
                return selected
        for state in ordered:
            if id(state) in selected_ids:
                continue
            selected.append(state)
            if len(selected) >= limit:
                break
        return selected

    def _select_indexed_state_subset(self, indexed_states, limit, prefer_pose_diversity=False):
        ordered = sorted(indexed_states, key=lambda item: item[1]["cost"])
        if any(item[1]["candidate"].get("approach_release", False) for item in ordered) and len(ordered) > limit:
            selected = []
            selected_state_ids = set()

            def add(item):
                state = item[1]
                if id(state) in selected_state_ids:
                    return False
                selected.append(item)
                selected_state_ids.add(id(state))
                return len(selected) >= limit

            add(ordered[0])
            seen_phase_keys = set()
            for item in ordered:
                candidate = item[1]["candidate"]
                if not candidate.get("approach_release", False):
                    continue
                key = self._approach_release_phase_key(candidate)
                if key in seen_phase_keys:
                    continue
                seen_phase_keys.add(key)
                if add(item):
                    return selected
                if len(seen_phase_keys) >= max(6, min(limit - 1, 12)):
                    break
            for item in ordered:
                if add(item):
                    return selected
            return selected
        if not prefer_pose_diversity or len(ordered) <= limit:
            return ordered[:limit]
        selected = []
        selected_state_ids = set()
        seen_pose_keys = set()
        for item in ordered:
            state = item[1]
            key = self._candidate_pose_key(state["candidate"])
            if key in seen_pose_keys:
                continue
            seen_pose_keys.add(key)
            selected.append(item)
            selected_state_ids.add(id(state))
            if len(selected) >= limit:
                return selected
        for item in ordered:
            state = item[1]
            if id(state) in selected_state_ids:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _candidate_failure_template(self, idx, ego_pose):
        return {
            "wp": idx,
            "pose": self._pose_xyz(ego_pose),
            "body_best_margin": float("-inf"),
            "body_obstacle": "none",
            "body_point": (0.0, 0.0, 0.0),
            "arm_best_margin": float("-inf"),
            "arm_obstacle": "none",
            "arm_point": (0.0, 0.0, 0.0),
            "pose_candidates": 0,
            "body_valid_candidates": 0,
            "shape_candidates": 0,
            "arm_valid_candidates": 0,
            "failure_reason": "no_candidate",
            "passage_risk": False,
            "passage_compact_lock": False,
            "passage_compact_safe": False,
            "passage_slot": "none",
            "release_factor": 0.0,
            "stable_shape": (0.0, 0.0),
            "q_ref": (0.0, 0.0),
        }

    def _make_sequence_candidate(
        self,
        pose,
        shape,
        ego_pose,
        desired_shape,
        body_margin,
        body_obstacle,
        body_point,
        arm_margin,
        arm_obstacle,
        arm_point,
        morphology_activation,
        passage_context=None,
    ):
        theta, phi = shape
        coupling = self._coupling_metrics(pose, theta, phi)
        if coupling["com_offset"] > self.hard_com_limit or coupling["gravity_torque"] > self.tau_limit:
            return None
        _, lateral_extent = self._arm_morphology_metrics(pose, theta, phi)
        candidate = {
            "pose": pose,
            "shape": (theta, phi),
            "ego_dev": point_distance(self._pose_xyz(ego_pose), self._pose_xyz(pose)),
            "body_margin": body_margin,
            "body_obstacle": body_obstacle,
            "body_point": body_point,
            "arm_margin": arm_margin,
            "arm_obstacle": arm_obstacle,
            "arm_point": arm_point,
            "coupling": coupling,
            "lateral_extent": lateral_extent,
            "morphology_activation": morphology_activation,
            "passage_risk": False,
            "passage_slot": "none",
            "passage_compact_safe": False,
            "passage_compact_lock": False,
            "release_factor": 0.0,
            "stable_shape": (self.compact_theta, self.compact_phi),
            "pre_reach_shape": (self.compact_theta, self.compact_phi),
            "q_ref": (self.compact_theta, self.compact_phi),
        }
        if passage_context:
            candidate.update(passage_context)
        candidate["node_cost"] = self._candidate_node_cost(
            candidate,
            candidate.get("q_ref", desired_shape),
            ego_pose,
            morphology_activation,
        )
        return candidate

    def _slot_altitude_targets(self, obstacle_name):
        if not obstacle_name:
            return []
        slot = None
        for item in self._slot_constraints():
            if (
                obstacle_name == item["id"]
                or obstacle_name == item["lower_name"]
                or obstacle_name == item["ceiling_name"]
            ):
                slot = item
                break
        if slot is None:
            return []
        z_low = slot["z_low"]
        z_high = slot["z_high"]
        mid = 0.5 * (z_low + z_high)
        vals = [
            z_high,
            slot["z_target"],
            z_high - 0.03,
            min(z_high, mid + 0.05),
            mid,
            max(z_low, mid - 0.03),
        ]
        out = []
        for value in vals:
            value = min(max(value, z_low), z_high)
            if all(abs(value - old) > 1e-3 for old in out):
                out.append(value)
        return out

    def _generate_sequence_candidates(self, ego_poses, idx, goal_theta, goal_phi, expanded=False, deadline=float("inf"), passage_schedule=None):
        ego_pose = ego_poses[idx]
        path_yaw = self._path_yaw_at(ego_poses, idx)
        desired_shape = self._desired_transit_shape(idx, len(ego_poses), goal_theta, goal_phi)
        lookahead_activation = self._lookahead_morphology_activation(ego_poses, idx)
        slot_constraint = self._slot_constraint_for_pose(ego_pose)
        scheduled_context = None
        if passage_schedule is not None and 0 <= idx < len(passage_schedule):
            scheduled_context = passage_schedule[idx]
        failure = self._candidate_failure_template(idx, ego_pose)
        candidates = []
        wp_candidate_target = self.sequence_candidates_per_wp
        arm_risk_target = max(self.sequence_risk_arm_margin, self.required_arm_margin)
        body_risk_target = max(self.sequence_risk_body_margin, self.required_body_margin)
        high_risk_slot = False
        terminal_final_approach = False
        terminal_approach_z_floor = self._pose_xyz(ego_poses[-1])[2] - 0.05
        base_passage_context = {
            "passage_risk": False,
            "passage_slot": "none",
            "passage_compact_safe": False,
            "passage_compact_lock": False,
            "release_factor": 0.0,
            "stable_shape": (self.compact_theta, self.compact_phi),
            "pre_reach_shape": (self.compact_theta, self.compact_phi),
            "q_ref": desired_shape,
            "passage_hold_shape": None,
            "passage_hold_segment_id": -1,
        }

        def finish(reason=None):
            if reason is not None:
                failure["failure_reason"] = reason
            return self._select_candidate_subset(candidates, wp_candidate_target, high_risk_slot or expanded), failure

        def context_for_pose(pose, pose_body_margin, compact_margin=None):
            if compact_margin is None:
                compact_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=5)
            context = self._passage_morphology_context(
                pose,
                pose_body_margin,
                compact_margin,
                idx,
                len(ego_poses),
                self._slot_constraint_for_pose(pose),
            )
            stable_shape = self._stable_reference_shape(pose)
            release_factor = self._morphology_release_factor(
                ego_poses,
                idx,
                context["passage_risk"],
                context.get("passage_slot", "none"),
            )
            pre_reach = (goal_theta, goal_phi)
            approach_release = bool(
                context.get("passage_slot", "none") == "none"
                and release_factor > 0.35
            )
            standoff_distance = point_distance(self._pose_xyz(pose), self._pose_xyz(ego_poses[-1]))
            terminal_pre_standoff = (
                idx == len(ego_poses) - 2
                and standoff_distance < 0.80
            )
            if approach_release and terminal_pre_standoff:
                # Do not unfold the arm while the body is still flying through
                # the final short approach into the selected standoff. Reach-in
                # is handled after the body reaches the selected standoff pose.
                release_factor = 0.0
                approach_release = False
                stable_shape = (self.compact_theta, self.compact_phi)
                pre_reach = (self.compact_theta, self.compact_phi)
            if scheduled_context is not None:
                scheduled_key = scheduled_context.get("segment_key", "")
                pose_slot = context.get("passage_slot", "none")
                same_slot = (
                    pose_slot != "none"
                    and (
                        scheduled_key == pose_slot
                        or scheduled_key == "%s_buffer" % pose_slot
                    )
                )
                use_scheduled_hold = (
                    scheduled_context.get("segment_buffer", False)
                    or same_slot
                    or (context["passage_risk"] and not approach_release)
                )
                if use_scheduled_hold:
                    stable_shape = scheduled_context["hold_shape"]
                    context["passage_hold_shape"] = scheduled_context["hold_shape"]
                    context["passage_hold_segment_id"] = scheduled_context["segment_id"]
                    context["passage_hold_segment_key"] = scheduled_context["segment_key"]
                    context["passage_hold_min_arm_margin"] = scheduled_context["hold_min_arm_margin"]
            q_ref = self._release_reference_shape(stable_shape, pre_reach, release_factor)
            context.update(
                {
                    "stable_shape": stable_shape,
                    "pre_reach_shape": pre_reach,
                    "q_ref": q_ref,
                    "release_factor": release_factor,
                    "approach_release": approach_release,
                }
            )
            return context

        body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(ego_pose))
        terminal_standoff_lock = idx == len(ego_poses) - 1
        failure["pose_candidates"] = 1
        failure["body_best_margin"] = body_margin
        failure["body_obstacle"] = body_obstacle
        failure["body_point"] = body_point
        if body_margin >= self.required_body_margin:
            failure["body_valid_candidates"] = 1
            free_shape = desired_shape if idx == 0 else (self.compact_theta, self.compact_phi)
            arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(ego_pose, free_shape[0], free_shape[1], sample_limit=7)
            failure["shape_candidates"] = 1
            failure["arm_best_margin"] = arm_margin
            failure["arm_obstacle"] = arm_obstacle
            failure["arm_point"] = arm_point
            compact_margin_for_context = arm_margin
            if abs(free_shape[0] - self.compact_theta) > 1e-6 or abs(self._angle_delta(free_shape[1], self.compact_phi)) > 1e-6:
                compact_margin_for_context = self._arm_margin(ego_pose, self.compact_theta, self.compact_phi, sample_limit=5)
            base_passage_context = context_for_pose(ego_pose, body_margin, compact_margin_for_context)
            failure["passage_risk"] = base_passage_context["passage_risk"]
            failure["passage_slot"] = base_passage_context["passage_slot"]
            failure["passage_compact_safe"] = base_passage_context["passage_compact_safe"]
            failure["passage_compact_lock"] = base_passage_context["passage_compact_lock"]
            failure["release_factor"] = base_passage_context["release_factor"]
            failure["approach_release"] = base_passage_context.get("approach_release", False)
            failure["stable_shape"] = base_passage_context["stable_shape"]
            failure["q_ref"] = base_passage_context["q_ref"]
            failure["passage_hold_shape"] = base_passage_context.get("passage_hold_shape")
            failure["passage_hold_segment_id"] = base_passage_context.get("passage_hold_segment_id", -1)
            desired_shape = base_passage_context["q_ref"]
            if terminal_standoff_lock and idx != 0:
                if arm_margin >= self.required_arm_margin:
                    terminal_context = dict(base_passage_context)
                    terminal_context.update(
                        {
                            "release_factor": 0.0,
                            "approach_release": False,
                            "stable_shape": (self.compact_theta, self.compact_phi),
                            "pre_reach_shape": (self.compact_theta, self.compact_phi),
                            "q_ref": (self.compact_theta, self.compact_phi),
                        }
                    )
                    failure["release_factor"] = terminal_context["release_factor"]
                    failure["approach_release"] = terminal_context["approach_release"]
                    failure["stable_shape"] = terminal_context["stable_shape"]
                    failure["q_ref"] = terminal_context["q_ref"]
                    candidate = self._make_sequence_candidate(
                        ego_pose,
                        (self.compact_theta, self.compact_phi),
                        ego_pose,
                        (self.compact_theta, self.compact_phi),
                        body_margin,
                        body_obstacle,
                        body_point,
                        arm_margin,
                        arm_obstacle,
                        arm_point,
                        0.0,
                        terminal_context,
                    )
                    if candidate is not None:
                        candidate["terminal_standoff_lock"] = True
                        failure["arm_valid_candidates"] = 1
                        return [candidate], failure
                failure["failure_reason"] = "terminal_standoff_lock_arm_margin"
                return [], failure
            focused_low_release = (
                base_passage_context["passage_risk"]
                and base_passage_context.get("release_factor", 0.0) < 0.15
            )
            high_risk_slot = (
                (slot_constraint is not None or arm_margin < arm_risk_target or body_margin < body_risk_target)
                and idx != len(ego_poses) - 1
            )
            low_release_slot = bool(high_risk_slot and focused_low_release)
            risk_margin = min(body_margin, arm_margin)
            moderate_margin = min(
                max(self.required_body_margin + 0.04, self.refine_clearance_target + 0.01),
                max(self.required_arm_margin + 0.04, self.refine_clearance_target + 0.01),
            )
            if risk_margin >= moderate_margin and idx != len(ego_poses) - 1:
                wp_candidate_target = min(self.sequence_candidates_per_wp, max(self.sequence_min_candidates_per_wp, 6))
            if high_risk_slot:
                if low_release_slot:
                    wp_candidate_target = max(self.sequence_min_candidates_per_wp, 6)
                else:
                    wp_candidate_target = max(self.sequence_candidates_per_wp, 24)
            if idx == len(ego_poses) - 1 and arm_margin >= self.required_arm_margin:
                candidate = self._make_sequence_candidate(
                    ego_pose,
                    free_shape,
                    ego_pose,
                    desired_shape,
                    body_margin,
                    body_obstacle,
                    body_point,
                    arm_margin,
                    arm_obstacle,
                    arm_point,
                    0.0,
                    base_passage_context,
                )
                if candidate is not None:
                    failure["arm_valid_candidates"] = 1
                    return [candidate], failure
            body_fast_margin = max(self.required_body_margin + 0.08, self.refine_clearance_target + 0.04)
            arm_fast_margin = max(self.required_arm_margin + 0.12, self.refine_clearance_target + 0.08)
            body_free_margin = max(self.required_body_margin + 0.04, self.refine_clearance_target + 0.01)
            arm_free_margin = max(self.required_arm_margin + 0.04, self.refine_clearance_target + 0.01)
            if (
                body_margin >= body_free_margin
                and arm_margin >= arm_free_margin
                and idx != len(ego_poses) - 1
            ):
                compact_candidate = self._make_sequence_candidate(
                    ego_pose,
                    free_shape,
                    ego_pose,
                    desired_shape,
                    body_margin,
                    body_obstacle,
                    body_point,
                    arm_margin,
                    arm_obstacle,
                    arm_point,
                    min(lookahead_activation, 0.5),
                    base_passage_context,
                )
                if (
                    compact_candidate is not None
                    and body_margin >= body_fast_margin
                    and arm_margin >= arm_fast_margin
                    and base_passage_context.get("release_factor", 0.0) < 0.10
                    and not expanded
                ):
                    failure["arm_valid_candidates"] = 1
                    return [compact_candidate], failure
                if (
                    compact_candidate is not None
                    and lookahead_activation <= 0.35
                    and base_passage_context.get("release_factor", 0.0) < 0.10
                    and not base_passage_context["passage_risk"]
                    and not expanded
                ):
                    failure["arm_valid_candidates"] = 1
                    return [compact_candidate], failure
                if compact_candidate is not None:
                    candidates.append(compact_candidate)
                    failure["arm_valid_candidates"] = 1
                    if terminal_final_approach:
                        return finish("terminal_compact_prepare")
                    if not focused_low_release:
                        prep_phis = [
                            self.compact_phi,
                            desired_shape[1],
                            -180.0,
                            -90.0,
                            0.0,
                            90.0,
                            180.0,
                        ]
                        prep_thetas = [
                            self.morphology_min_active_theta,
                            max(self.morphology_min_active_theta, 60.0),
                            max(self.morphology_min_active_theta, 90.0),
                        ]
                        seen_prep = set()
                        safe_prepare_target = 4 if not expanded else 6
                        for theta in prep_thetas:
                            for phi in prep_phis:
                                key = (round(theta, 3), round(self._normalize_angle_deg(phi), 3))
                                if key in seen_prep:
                                    continue
                                seen_prep.add(key)
                                failure["shape_candidates"] += 1
                                test_phi = self._normalize_angle_deg(phi)
                                fast_margin, fast_obstacle, fast_point = self._arm_margin_detail(ego_pose, theta, test_phi, sample_limit=7)
                                if fast_margin < self.required_arm_margin:
                                    continue
                                full_margin, full_obstacle, full_point = self._arm_margin_detail(ego_pose, theta, test_phi)
                                if full_margin < self.required_arm_margin:
                                    continue
                                candidate = self._make_sequence_candidate(
                                    ego_pose,
                                    (theta, test_phi),
                                    ego_pose,
                                    desired_shape,
                                    body_margin,
                                    body_obstacle,
                                    body_point,
                                    full_margin,
                                    full_obstacle,
                                    full_point,
                                    lookahead_activation,
                                    base_passage_context,
                                )
                                if candidate is None:
                                    continue
                                candidates.append(candidate)
                                failure["arm_valid_candidates"] += 1
                        if candidates and not expanded:
                            return finish("safe_prepare_candidates")
        if terminal_standoff_lock and idx != 0:
            failure["failure_reason"] = "terminal_standoff_lock_body_margin"
            return [], failure
        high_risk_slot = (
            (slot_constraint is not None or failure["arm_best_margin"] < arm_risk_target or failure["body_best_margin"] < body_risk_target)
            and idx != len(ego_poses) - 1
        )
        if terminal_final_approach:
            high_risk_slot = True
        early_return_target = self.sequence_min_candidates_per_wp
        if high_risk_slot or expanded:
            early_return_target = min(
                wp_candidate_target,
                max(self.sequence_min_candidates_per_wp * 2, 10),
            )
            if base_passage_context.get("passage_risk", False) and base_passage_context.get("release_factor", 0.0) < 0.15:
                early_return_target = min(wp_candidate_target, max(self.sequence_min_candidates_per_wp, 6))
        if high_risk_slot and expanded:
            early_return_target = min(
                wp_candidate_target,
                max(early_return_target, 14),
            )
        min_pose_diversity_for_early_return = 0
        if expanded:
            min_pose_diversity_for_early_return = 3
        if high_risk_slot:
            min_pose_diversity_for_early_return = 4
            if expanded:
                min_pose_diversity_for_early_return = 6
            if base_passage_context.get("passage_risk", False) and base_passage_context.get("release_factor", 0.0) < 0.15:
                min_pose_diversity_for_early_return = 2

        def enough_pose_diversity_for_early_return():
            if min_pose_diversity_for_early_return <= 0:
                return True
            return self._candidate_pose_diversity(candidates) >= min_pose_diversity_for_early_return

        def enough_candidates_for_early_return(target):
            return len(candidates) >= target and enough_pose_diversity_for_early_return()

        lat_scale = 2.0 if expanded else 1.0
        z_scale = 1.5 if expanded else 1.0
        if high_risk_slot:
            lateral_offsets = [
                -1.5 * self.refine_lateral_step,
                -1.0 * self.refine_lateral_step,
                -0.5 * self.refine_lateral_step,
                0.0,
                0.5 * self.refine_lateral_step,
                1.0 * self.refine_lateral_step,
                1.5 * self.refine_lateral_step,
            ]
            z_offsets = [
                -4.0 * self.refine_z_step,
                -3.0 * self.refine_z_step,
                -2.0 * self.refine_z_step,
                -1.5 * self.refine_z_step,
                -1.0 * self.refine_z_step,
                -0.5 * self.refine_z_step,
                0.0,
                0.5 * self.refine_z_step,
                1.0 * self.refine_z_step,
                1.5 * self.refine_z_step,
                2.0 * self.refine_z_step,
                2.5 * self.refine_z_step,
                3.0 * self.refine_z_step,
                3.5 * self.refine_z_step,
                4.0 * self.refine_z_step,
                4.5 * self.refine_z_step,
                5.0 * self.refine_z_step,
                6.0 * self.refine_z_step,
                7.0 * self.refine_z_step,
            ]
            if terminal_final_approach:
                z_offsets = [
                    0.0,
                    2.0 * self.refine_z_step,
                    3.0 * self.refine_z_step,
                    4.0 * self.refine_z_step,
                    5.0 * self.refine_z_step,
                    6.0 * self.refine_z_step,
                    7.0 * self.refine_z_step,
                    8.0 * self.refine_z_step,
                    9.0 * self.refine_z_step,
                ]
        else:
            lateral_offsets = [
                -2.0 * lat_scale * self.refine_lateral_step,
                -1.0 * lat_scale * self.refine_lateral_step,
                0.0,
                1.0 * lat_scale * self.refine_lateral_step,
                2.0 * lat_scale * self.refine_lateral_step,
            ]
            z_offsets = [
                -0.5 * z_scale * self.refine_z_step,
                0.0,
                0.5 * z_scale * self.refine_z_step,
                1.0 * z_scale * self.refine_z_step,
                1.5 * z_scale * self.refine_z_step,
                2.0 * z_scale * self.refine_z_step,
                3.0 * z_scale * self.refine_z_step,
                4.0 * z_scale * self.refine_z_step,
                5.0 * z_scale * self.refine_z_step,
            ]
        yaw_offsets = [-self.refine_yaw_step_deg, 0.0, self.refine_yaw_step_deg]
        if idx == len(ego_poses) - 1:
            pose_offsets = [(0.0, 0.0)]
            yaw_offsets = [0.0]
        elif high_risk_slot:
            pose_offsets = sorted(
                [(lat, dz) for lat in lateral_offsets for dz in z_offsets],
                key=lambda item: (
                    abs(item[0]),
                    abs(item[1]),
                    -item[1],
                ),
            )
        else:
            pose_offsets = self._ordered_pose_offsets(lateral_offsets, z_offsets)
            yaw_offsets = self._ordered_yaw_offsets(yaw_offsets)
        pose_pool = []
        pose_test_limit = self.sequence_max_pose_tests_per_wp * (2 if expanded else 1)

        def terminal_adjust_pose(pose):
            if not terminal_final_approach:
                return pose
            x, y, z = self._pose_xyz(pose)
            if z >= terminal_approach_z_floor:
                return pose
            return self._make_pose(x, y, terminal_approach_z_floor, self._pose_yaw(pose))

        if high_risk_slot:
            pose_test_limit = max(pose_test_limit, len(pose_offsets) * len(yaw_offsets))
            base_xyz = self._pose_xyz(ego_pose)
            left = (-math.sin(path_yaw), math.cos(path_yaw))
            slot_targets = self._slot_altitude_targets(failure.get("arm_obstacle", ""))
            if slot_constraint is not None:
                slot_targets = self._slot_altitude_targets(slot_constraint["id"]) + slot_targets
            low_release_slot = bool(
                base_passage_context.get("passage_risk", False)
                and base_passage_context.get("release_factor", 0.0) < 0.15
            )
            slot_laterals = [0.0] if low_release_slot else [0.0, -0.5 * self.refine_lateral_step, 0.5 * self.refine_lateral_step]
            if expanded:
                slot_laterals.extend(
                    [
                        -0.5 * self.refine_lateral_step,
                        0.5 * self.refine_lateral_step,
                    ]
                    if low_release_slot
                    else [
                        -1.0 * self.refine_lateral_step,
                        1.0 * self.refine_lateral_step,
                        -1.5 * self.refine_lateral_step,
                        1.5 * self.refine_lateral_step,
                    ]
                )
            seen_slot_targets = []
            for target_z in slot_targets:
                if all(abs(target_z - old) > 1e-3 for old in seen_slot_targets):
                    seen_slot_targets.append(target_z)
            for target_z in seen_slot_targets:
                for lateral in slot_laterals:
                    if time.monotonic() >= deadline:
                        failure["failure_reason"] = "partial_timeout" if candidates else "timeout"
                        return finish(failure["failure_reason"])
                    pose = self._make_pose(
                        base_xyz[0] + left[0] * lateral,
                        base_xyz[1] + left[1] * lateral,
                        target_z,
                        path_yaw,
                    )
                    pose = terminal_adjust_pose(pose)
                    failure["pose_candidates"] += 1
                    body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
                    if body_margin > failure["body_best_margin"]:
                        failure["body_best_margin"] = body_margin
                        failure["body_obstacle"] = body_obstacle
                        failure["body_point"] = body_point
                    if body_margin < self.required_body_margin:
                        continue
                    failure["body_valid_candidates"] += 1
                    compact_margin, compact_obstacle, compact_point = self._arm_margin_detail(
                        pose,
                        self.compact_theta,
                        self.compact_phi,
                        sample_limit=5,
                    )
                    if compact_margin > failure["arm_best_margin"]:
                        failure["arm_best_margin"] = compact_margin
                        failure["arm_obstacle"] = compact_obstacle
                        failure["arm_point"] = compact_point
                    pose_passage_context = context_for_pose(pose, body_margin, compact_margin)
                    ego_dev = point_distance(base_xyz, self._pose_xyz(pose))
                    body_deficit = max(0.0, body_risk_target - body_margin) / max(body_risk_target, 1e-3)
                    arm_deficit = max(0.0, arm_risk_target - compact_margin) / max(arm_risk_target, 1e-3)
                    pose_score = (
                        0.4 * ego_dev
                        + 10.0 * body_deficit * body_deficit
                        + 8.0 * arm_deficit * arm_deficit
                        - 1.5 * min(body_margin, body_risk_target)
                        - 1.2 * min(compact_margin, arm_risk_target)
                    )
                    pose_pool.append((pose_score, pose, body_margin, body_obstacle, body_point))
                    if terminal_final_approach:
                        direct_shapes = [(self.compact_theta, self.compact_phi)]
                    elif pose_passage_context["passage_risk"] and pose_passage_context.get("release_factor", 0.0) < 0.15:
                        direct_shapes = self._focused_stable_shape_candidates(pose_passage_context)
                    else:
                        direct_shapes = [(self.compact_theta, self.compact_phi)] + self._slot_shape_seeds(pose)
                    seen_direct_shapes = set()
                    for shape in direct_shapes:
                        key = (round(shape[0], 3), round(self._normalize_angle_deg(shape[1]), 3))
                        if key in seen_direct_shapes:
                            continue
                        seen_direct_shapes.add(key)
                        if time.monotonic() >= deadline:
                            failure["failure_reason"] = "partial_timeout" if candidates else "timeout"
                            return finish(failure["failure_reason"])
                        activation = 0.5 if abs(shape[0] - self.compact_theta) < 1e-3 else 1.0
                        self._try_add_sequence_candidate(
                            candidates,
                            failure,
                            pose,
                            (shape[0], self._normalize_angle_deg(shape[1])),
                            ego_pose,
                            desired_shape,
                            body_margin,
                            body_obstacle,
                            body_point,
                            max(lookahead_activation, activation),
                            pose_passage_context,
                        )
                        if enough_candidates_for_early_return(early_return_target):
                            return finish("slot_direct_candidates")
        for lateral, dz in pose_offsets:
            for yaw_offset in yaw_offsets:
                if time.monotonic() >= deadline:
                    failure["failure_reason"] = "partial_timeout" if candidates else "timeout"
                    return finish(failure["failure_reason"])
                if failure["pose_candidates"] >= pose_test_limit:
                    break
                pose = self._copy_pose(ego_pose) if idx == len(ego_poses) - 1 else self._offset_pose(ego_pose, path_yaw, lateral, dz, yaw_offset)
                pose = terminal_adjust_pose(pose)
                failure["pose_candidates"] += 1
                body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
                if body_margin > failure["body_best_margin"]:
                    failure["body_best_margin"] = body_margin
                    failure["body_obstacle"] = body_obstacle
                    failure["body_point"] = body_point
                if body_margin < self.required_body_margin:
                    failure["failure_reason"] = "body_margin"
                    continue
                failure["body_valid_candidates"] += 1
                ego_dev = point_distance(self._pose_xyz(ego_pose), self._pose_xyz(pose))
                if high_risk_slot:
                    compact_margin, compact_obstacle, compact_point = self._arm_margin_detail(
                        pose,
                        self.compact_theta,
                        self.compact_phi,
                        sample_limit=5,
                    )
                    if compact_margin > failure["arm_best_margin"]:
                        failure["arm_best_margin"] = compact_margin
                        failure["arm_obstacle"] = compact_obstacle
                        failure["arm_point"] = compact_point
                    body_deficit = max(0.0, body_risk_target - body_margin) / max(body_risk_target, 1e-3)
                    arm_deficit = max(0.0, arm_risk_target - compact_margin) / max(arm_risk_target, 1e-3)
                    pose_score = (
                        ego_dev
                        + 10.0 * body_deficit * body_deficit
                        + 8.0 * arm_deficit * arm_deficit
                        - 1.5 * min(body_margin, body_risk_target)
                        - 1.2 * min(compact_margin, arm_risk_target)
                        + 0.04 * abs(dz)
                    )
                else:
                    pose_score = ego_dev - 0.2 * min(body_margin, self.refine_clearance_target)
                pose_pool.append((pose_score, pose, body_margin, body_obstacle, body_point))
            if failure["pose_candidates"] >= pose_test_limit:
                break
        pose_pool.sort(key=lambda item: item[0])
        pose_limit = self.sequence_max_body_poses_per_wp * (2 if expanded else 1)
        if high_risk_slot:
            if base_passage_context.get("passage_risk", False) and base_passage_context.get("release_factor", 0.0) < 0.15:
                pose_limit = max(pose_limit, 8)
            else:
                pose_limit = max(pose_limit, 20)
        pose_pool = pose_pool[: pose_limit]
        shape_tests = 0
        shape_test_budget = self.sequence_max_shape_tests_per_wp * (2 if expanded else 1)
        if high_risk_slot:
            if base_passage_context.get("passage_risk", False) and base_passage_context.get("release_factor", 0.0) < 0.15:
                shape_test_budget = min(max(shape_test_budget, 80), 160)
            else:
                shape_test_budget = max(shape_test_budget, 900)
        for _, pose, body_margin, body_obstacle, body_point in pose_pool:
                if time.monotonic() >= deadline:
                    failure["failure_reason"] = "partial_timeout" if candidates else "timeout"
                    return finish(failure["failure_reason"])
                home_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=5)
                morphology_activation = max(self._morphology_activation(home_margin, False), lookahead_activation)
                pose_passage_context = context_for_pose(pose, body_margin, home_margin)
                if high_risk_slot:
                    compact_margin, compact_obstacle, compact_point = self._arm_margin_detail(
                        pose,
                        self.compact_theta,
                        self.compact_phi,
                    )
                    if compact_margin > failure["arm_best_margin"]:
                        failure["arm_best_margin"] = compact_margin
                        failure["arm_obstacle"] = compact_obstacle
                        failure["arm_point"] = compact_point
                    if compact_margin >= self.required_arm_margin:
                        failure["shape_candidates"] += 1
                        candidate = self._make_sequence_candidate(
                            pose,
                            (self.compact_theta, self.compact_phi),
                            ego_pose,
                            desired_shape,
                            body_margin,
                            body_obstacle,
                            body_point,
                            compact_margin,
                            compact_obstacle,
                            compact_point,
                            min(morphology_activation, 0.5),
                            pose_passage_context,
                        )
                        if candidate is not None:
                            candidates.append(candidate)
                            failure["arm_valid_candidates"] += 1
                            if enough_candidates_for_early_return(early_return_target):
                                return finish("high_risk_compact_lift")
                    if not (pose_passage_context["passage_risk"] and pose_passage_context.get("release_factor", 0.0) < 0.15):
                        for theta, phi in self._slot_shape_seeds(pose):
                            if time.monotonic() >= deadline:
                                failure["failure_reason"] = "partial_timeout" if candidates else "timeout"
                                return finish(failure["failure_reason"])
                            failure["shape_candidates"] += 1
                            arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(pose, theta, phi)
                            if arm_margin > failure["arm_best_margin"]:
                                failure["arm_best_margin"] = arm_margin
                                failure["arm_obstacle"] = arm_obstacle
                                failure["arm_point"] = arm_point
                            if arm_margin < self.required_arm_margin:
                                continue
                            candidate = self._make_sequence_candidate(
                                pose,
                                (theta, phi),
                                ego_pose,
                                desired_shape,
                                body_margin,
                                body_obstacle,
                                body_point,
                                arm_margin,
                                arm_obstacle,
                                arm_point,
                                max(morphology_activation, 1.0),
                                pose_passage_context,
                            )
                            if candidate is None:
                                continue
                            candidates.append(candidate)
                            failure["arm_valid_candidates"] += 1
                            if enough_candidates_for_early_return(early_return_target):
                                return finish("slot_shape_seed")
                if pose_passage_context["passage_risk"] and pose_passage_context.get("release_factor", 0.0) < 0.15:
                    if terminal_final_approach:
                        shape_sequence = [(self.compact_theta, self.compact_phi)]
                    else:
                        shape_sequence = self._focused_stable_shape_candidates(pose_passage_context)
                else:
                    shape_sequence = self._shape_candidates_for_sequence(
                        idx,
                        len(ego_poses),
                        pose_passage_context.get("q_ref", desired_shape),
                        morphology_activation,
                        pose_passage_context,
                    )
                for theta, phi in shape_sequence:
                    if time.monotonic() >= deadline:
                        failure["failure_reason"] = "partial_timeout" if len(candidates) >= self.sequence_min_candidates_per_wp else "timeout"
                        return finish(failure["failure_reason"])
                    shape_tests += 1
                    if shape_tests > shape_test_budget:
                        failure["failure_reason"] = "shape_test_budget"
                        break
                    if (
                        morphology_activation > 0.5
                        and idx != len(ego_poses) - 1
                        and not pose_passage_context.get("approach_release", False)
                        and abs(theta - self.compact_theta) < self.morphology_min_active_theta
                    ):
                        continue
                    failure["shape_candidates"] += 1
                    fast_arm_margin, fast_arm_obstacle, fast_arm_point = self._arm_margin_detail(pose, theta, phi, sample_limit=5)
                    if fast_arm_margin < self.required_arm_margin:
                        if fast_arm_margin > failure["arm_best_margin"]:
                            failure["arm_best_margin"] = fast_arm_margin
                            failure["arm_obstacle"] = fast_arm_obstacle
                            failure["arm_point"] = fast_arm_point
                        failure["failure_reason"] = "arm_margin"
                        continue
                    arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(pose, theta, phi)
                    if arm_margin > failure["arm_best_margin"]:
                        failure["arm_best_margin"] = arm_margin
                        failure["arm_obstacle"] = arm_obstacle
                        failure["arm_point"] = arm_point
                    if arm_margin < self.required_arm_margin:
                        failure["failure_reason"] = "arm_margin"
                        continue
                    coupling = self._coupling_metrics(pose, theta, phi)
                    if coupling["com_offset"] > self.hard_com_limit:
                        failure["failure_reason"] = "com_limit"
                        continue
                    if coupling["gravity_torque"] > self.tau_limit:
                        failure["failure_reason"] = "torque_limit"
                        continue
                    _, lateral_extent = self._arm_morphology_metrics(pose, theta, phi)
                    failure["arm_valid_candidates"] += 1
                    candidate = {
                        "pose": pose,
                        "shape": (theta, phi),
                        "ego_dev": point_distance(self._pose_xyz(ego_pose), self._pose_xyz(pose)),
                        "body_margin": body_margin,
                        "body_obstacle": body_obstacle,
                        "body_point": body_point,
                        "arm_margin": arm_margin,
                        "arm_obstacle": arm_obstacle,
                        "arm_point": arm_point,
                        "coupling": coupling,
                        "lateral_extent": lateral_extent,
                        "morphology_activation": morphology_activation,
                        "passage_risk": pose_passage_context["passage_risk"],
                        "passage_slot": pose_passage_context["passage_slot"],
                        "passage_compact_safe": pose_passage_context["passage_compact_safe"],
                        "passage_compact_lock": pose_passage_context["passage_compact_lock"],
                        "release_factor": pose_passage_context["release_factor"],
                        "stable_shape": pose_passage_context["stable_shape"],
                        "pre_reach_shape": pose_passage_context["pre_reach_shape"],
                        "q_ref": pose_passage_context["q_ref"],
                    }
                    candidate["node_cost"] = self._candidate_node_cost(
                        candidate,
                        pose_passage_context.get("q_ref", desired_shape),
                        ego_pose,
                        morphology_activation,
                    )
                    candidates.append(candidate)
                    if len(candidates) >= self.sequence_min_candidates_per_wp and time.monotonic() + 0.05 >= deadline:
                        return finish("anytime_budget")
                    if enough_candidates_for_early_return(wp_candidate_target):
                        return finish("enough_candidates")
                if enough_candidates_for_early_return(wp_candidate_target) or shape_tests > shape_test_budget:
                    break
        return finish()

    def _sequence_transition_cost(self, prev, curr, prev_prev=None, include_safety=True):
        risk_edge = include_safety and self.swept_validation_enabled and self._sequence_edge_dense_check_needed(prev, curr)
        edge_safety_penalty = self._sequence_edge_safety_penalty(prev, curr) if risk_edge else 0.0
        if (
            risk_edge
            and self._sequence_edge_required_unsafe(prev, curr)
        ):
            return None
        dt = self._segment_dt_for_motion(prev["pose"], curr["pose"], prev["shape"], curr["shape"])
        # The final trajectory has a time-allocation refinement stage. Do not
        # prune an otherwise valid morphology transition only because the
        # initial nominal timing is too aggressive; final validation below still
        # enforces the actual rate/acceleration limits after time scaling.
        limit_dt = self._segment_dt_for_motion(prev["pose"], curr["pose"], prev["shape"], curr["shape"], self.time_refine_max_scale)
        dtheta = abs(curr["shape"][0] - prev["shape"][0])
        dphi = abs(self._angle_delta(prev["shape"][1], curr["shape"][1]))
        theta_scale = min(
            1.0,
            max(abs(prev["shape"][0]), abs(curr["shape"][0]))
            / max(float(self.arm_params["theta_max_deg"]), 1.0),
        )
        effective_dphi = dphi * max(0.15, theta_scale)
        theta_dot = dtheta / max(limit_dt, 1e-3)
        phi_dot = effective_dphi / max(limit_dt, 1e-3)
        dist = point_distance(self._pose_xyz(prev["pose"]), self._pose_xyz(curr["pose"]))
        yaw_delta = abs(math.atan2(math.sin(self._pose_yaw(curr["pose"]) - self._pose_yaw(prev["pose"])), math.cos(self._pose_yaw(curr["pose"]) - self._pose_yaw(prev["pose"]))))
        yaw_rate = yaw_delta / max(limit_dt, 1e-3)
        com_change = abs(curr["coupling"]["com_offset"] - prev["coupling"]["com_offset"])
        torque_change = abs(curr["coupling"]["gravity_torque"] - prev["coupling"]["gravity_torque"])
        lateral_change = abs(float(curr.get("lateral_extent", 0.0)) - float(prev.get("lateral_extent", 0.0)))
        release_jump = abs(float(curr.get("release_factor", 0.0)) - float(prev.get("release_factor", 0.0)))
        if (
            include_safety
            and not risk_edge
            and self.swept_validation_enabled
            and (
                prev.get("passage_risk", False)
                or curr.get("passage_risk", False)
                or release_jump > 0.15
                or dtheta > 25.0
                or dphi > 60.0
            )
        ):
            edge_safety_penalty += 0.45 * self._sequence_edge_safety_penalty(prev, curr)
        theta_rate_excess = max(0.0, theta_dot / max(self.theta_dot_max, 1e-3) - 1.0)
        phi_rate_excess = max(0.0, phi_dot / max(self.phi_dot_max, 1e-3) - 1.0)
        yaw_rate_excess = max(0.0, yaw_rate / max(self.yaw_rate_max, 1e-3) - 1.0)
        cost = (
            0.8 * dist
            + 0.035 * dtheta
            + 0.012 * effective_dphi
            + 1.5 * com_change
            + 0.15 * torque_change
            + 1.2 * lateral_change
            + self.morphology_release_transition_weight * release_jump * release_jump
            + 0.4 * yaw_rate
            + 6.0 * theta_rate_excess * theta_rate_excess
            + 4.0 * phi_rate_excess * phi_rate_excess
            + 4.0 * yaw_rate_excess * yaw_rate_excess
            + edge_safety_penalty
        )
        if prev.get("passage_risk", False) or curr.get("passage_risk", False):
            shape_jump = (dtheta / 25.0) ** 2 + (effective_dphi / 55.0) ** 2
            cost += self.passage_transition_shape_weight * shape_jump
        if prev_prev is not None:
            prev_dt = self._segment_dt_for_motion(prev_prev["pose"], prev["pose"], prev_prev["shape"], prev["shape"])
            prev_limit_dt = self._segment_dt_for_motion(prev_prev["pose"], prev["pose"], prev_prev["shape"], prev["shape"], self.time_refine_max_scale)
            prev_theta_dot = abs(prev["shape"][0] - prev_prev["shape"][0]) / max(prev_limit_dt, 1e-3)
            prev_dphi = abs(self._angle_delta(prev_prev["shape"][1], prev["shape"][1]))
            prev_theta_scale = min(
                1.0,
                max(abs(prev_prev["shape"][0]), abs(prev["shape"][0]))
                / max(float(self.arm_params["theta_max_deg"]), 1.0),
            )
            prev_phi_dot = (prev_dphi * max(0.15, prev_theta_scale)) / max(prev_limit_dt, 1e-3)
            theta_acc = abs(theta_dot - prev_theta_dot) / max(0.5 * (limit_dt + prev_limit_dt), 1e-3)
            phi_acc = abs(phi_dot - prev_phi_dot) / max(0.5 * (limit_dt + prev_limit_dt), 1e-3)
            theta_acc_excess = max(0.0, theta_acc / max(self.theta_acc_max, 1e-3) - 1.0)
            phi_acc_excess = max(0.0, phi_acc / max(self.phi_acc_max, 1e-3) - 1.0)
            cost += (
                0.02 * theta_acc
                + 0.008 * phi_acc
                + 4.0 * theta_acc_excess * theta_acc_excess
                + 3.0 * phi_acc_excess * phi_acc_excess
            )
        return cost

    def _edge_repair_pose_ladders(self, prev_pose, curr_pose):
        ax, ay, az = self._pose_xyz(prev_pose)
        bx, by, bz = self._pose_xyz(curr_pose)
        yaw = math.atan2(by - ay, bx - ax)
        left = (-math.sin(yaw), math.cos(yaw))
        slot_ladder = self._slot_projected_edge_ladder(prev_pose, curr_pose, min_knots=2, max_knots=3)
        if slot_ladder:
            return [slot_ladder]
        offsets = []
        lateral_values = [
            0.0,
            self.refine_lateral_step,
            -self.refine_lateral_step,
            2.0 * self.refine_lateral_step,
            -2.0 * self.refine_lateral_step,
            3.0 * self.refine_lateral_step,
            -3.0 * self.refine_lateral_step,
            4.0 * self.refine_lateral_step,
            -4.0 * self.refine_lateral_step,
        ]
        z_values = [
            0.0,
            self.refine_z_step,
            2.0 * self.refine_z_step,
            3.0 * self.refine_z_step,
            4.0 * self.refine_z_step,
            -self.refine_z_step,
            -2.0 * self.refine_z_step,
        ]
        for lateral in lateral_values:
            for dz in z_values:
                offsets.append((lateral, dz))
        offsets.sort(key=lambda item: (abs(item[0]) + 0.8 * abs(item[1]) - 0.25 * max(item[1], 0.0), abs(item[0])))

        ladders = []
        seen = set()
        ratio_sets = [(0.5,)]
        if self.edge_repair_max_knots >= 2:
            ratio_sets.extend([(1.0 / 3.0, 2.0 / 3.0), (0.4, 0.7), (0.3, 0.6)])
        for ratios in ratio_sets:
            for lateral, dz in offsets:
                poses = []
                key_parts = []
                for ratio in ratios:
                    base = self._interpolate_pose_between(prev_pose, curr_pose, ratio)
                    x, y, z = self._pose_xyz(base)
                    pose = self._make_pose(
                        x + left[0] * lateral,
                        y + left[1] * lateral,
                        max(0.90, min(2.05, z + dz)),
                        yaw,
                    )
                    pose, _, _ = self._project_pose_to_slot_window(pose)
                    poses.append(pose)
                    key_parts.append(
                        (
                            round(float(pose.position.x), 2),
                            round(float(pose.position.y), 2),
                            round(float(pose.position.z), 2),
                        )
                    )
                key = tuple(key_parts)
                if key in seen:
                    continue
                seen.add(key)
                ladders.append(poses)
                if len(ladders) >= self.edge_repair_max_ladders:
                    return ladders
        return ladders

    def _terminal_edge_repair_pose_ladders(self, prev_cand, curr_cand):
        prev_pose = prev_cand["pose"]
        curr_pose = curr_cand["pose"]
        ax, ay, az = self._pose_xyz(prev_pose)
        bx, by, bz = self._pose_xyz(curr_pose)
        yaw = math.atan2(by - ay, bx - ax)
        left = (-math.sin(yaw), math.cos(yaw))
        terminal_z = max(0.90, min(2.05, bz - 0.03))
        high_z = max(terminal_z, min(2.05, bz + 0.12))
        ratio_sets = [
            (0.28, 0.62, 0.86),
            (0.35, 0.68, 0.88),
            (0.42, 0.74),
            (0.50, 0.82),
        ]
        lateral_sets = [
            (0.0, 0.0, 0.0),
            (0.25, 0.25, 0.0),
            (-0.25, -0.25, 0.0),
            (0.45, 0.45, 0.0),
            (-0.45, -0.45, 0.0),
            (0.65, 0.50, 0.0),
            (-0.65, -0.50, 0.0),
            (0.85, 0.60, 0.0),
            (-0.85, -0.60, 0.0),
        ]
        z_profiles = [
            "low_then_terminal",
            "low_side_then_high",
            "smooth_rise",
            "high_side",
        ]
        ladders = []
        seen = set()
        for ratios in ratio_sets:
            for lateral_values in lateral_sets:
                for z_profile in z_profiles:
                    poses = []
                    key_parts = []
                    for i, ratio in enumerate(ratios):
                        base = self._interpolate_pose_between(prev_pose, curr_pose, ratio)
                        x, y, z = self._pose_xyz(base)
                        lateral = lateral_values[min(i, len(lateral_values) - 1)]
                        if z_profile == "low_then_terminal":
                            target_z = az + max(0.0, ratio - 0.55) / 0.45 * (terminal_z - az)
                        elif z_profile == "low_side_then_high":
                            target_z = az + (0.04 if i == 0 else max(0.0, ratio - 0.45) / 0.55 * (high_z - az))
                        elif z_profile == "smooth_rise":
                            target_z = az + ratio * (terminal_z - az)
                        else:
                            target_z = high_z if i < len(ratios) - 1 else terminal_z
                        pose = self._make_pose(
                            x + left[0] * lateral,
                            y + left[1] * lateral,
                            max(0.90, min(2.05, target_z)),
                            yaw,
                        )
                        pose, _, _ = self._project_pose_to_slot_window(pose)
                        poses.append(pose)
                        key_parts.append(
                            (
                                round(float(pose.position.x), 2),
                                round(float(pose.position.y), 2),
                                round(float(pose.position.z), 2),
                            )
                        )
                    key = tuple(key_parts)
                    if key in seen:
                        continue
                    seen.add(key)
                    ladders.append(poses)
                    for knot_idx in range(len(poses)):
                        for local_lat in (-0.12, -0.06, 0.06, 0.12):
                            for local_dz in (-0.04, -0.02, 0.02, 0.04, 0.07):
                                local_poses = [self._copy_pose(pose) for pose in poses]
                                px, py, pz = self._pose_xyz(local_poses[knot_idx])
                                local_poses[knot_idx] = self._make_pose(
                                    px + left[0] * local_lat,
                                    py + left[1] * local_lat,
                                    max(0.90, min(2.05, pz + local_dz)),
                                    self._pose_yaw(local_poses[knot_idx]),
                                )
                                local_poses[knot_idx], _, _ = self._project_pose_to_slot_window(local_poses[knot_idx])
                                local_key = tuple(
                                    (
                                        round(float(pose.position.x), 2),
                                        round(float(pose.position.y), 2),
                                        round(float(pose.position.z), 2),
                                    )
                                    for pose in local_poses
                                )
                                if local_key in seen:
                                    continue
                                seen.add(local_key)
                                ladders.append(local_poses)
        return ladders

    def _edge_repair_candidates_for_pose(self, pose, prev_cand, curr_cand, ratio, terminal_bridge=False):
        body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
        if body_margin < self.required_body_margin:
            return []
        prev_shape = prev_cand["shape"]
        curr_shape = curr_cand["shape"]
        mid_shape = (
            prev_shape[0] + (curr_shape[0] - prev_shape[0]) * ratio,
            self._normalize_angle_deg(prev_shape[1] + self._angle_delta(prev_shape[1], curr_shape[1]) * ratio),
        )
        prev_release = float(prev_cand.get("release_factor", 0.0))
        curr_release = float(curr_cand.get("release_factor", 0.0))
        release_factor = prev_release + (curr_release - prev_release) * ratio
        passage_risk = bool(prev_cand.get("passage_risk", False) or curr_cand.get("passage_risk", False) or self._slot_constraint_for_pose(pose) is not None)
        stable_shape = self._blend_shape(
            prev_cand.get("stable_shape", prev_shape),
            curr_cand.get("stable_shape", curr_shape),
            ratio,
        )
        q_ref = self._blend_shape(
            prev_cand.get("q_ref", prev_shape),
            curr_cand.get("q_ref", curr_shape),
            ratio,
        )
        compact_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=5)
        passage_context = self._passage_morphology_context(
            pose,
            body_margin,
            compact_margin,
            0,
            1,
            self._slot_constraint_for_pose(pose),
        )
        passage_context.update(
            {
                "passage_risk": passage_risk or passage_context["passage_risk"],
                "release_factor": release_factor,
                "stable_shape": stable_shape,
                "pre_reach_shape": curr_cand.get("pre_reach_shape", curr_shape),
                "q_ref": q_ref,
                "approach_release": bool(
                    (prev_cand.get("approach_release", False) or curr_cand.get("approach_release", False))
                    and release_factor > 0.25
                ),
            }
        )
        low_release_passage = passage_context["passage_risk"] and release_factor < 0.15
        if passage_context.get("approach_release", False):
            shape_seeds = [
                mid_shape,
                prev_shape,
                curr_shape,
            ]
            shape_seeds.extend(
                self._release_bridge_shape_candidates(
                    stable_shape,
                    passage_context.get("pre_reach_shape", curr_shape),
                    q_ref,
                )
            )
        elif low_release_passage:
            theta_min = float(self.arm_params["theta_min_deg"])
            theta_max = float(self.arm_params["theta_max_deg"])
            low_theta = min(max(theta_min, 0.0), theta_max)
            active_low_theta = min(max(self.morphology_min_active_theta, theta_min), theta_max)
            shape_seeds = self._focused_stable_shape_candidates(passage_context)
            shape_seeds = [
                mid_shape,
                prev_shape,
                curr_shape,
                stable_shape,
                q_ref,
                (low_theta, stable_shape[1]),
                (low_theta, q_ref[1]),
                (active_low_theta, stable_shape[1]),
                (active_low_theta, q_ref[1]),
            ] + shape_seeds
        else:
            shape_seeds = [
                mid_shape,
                prev_shape,
                curr_shape,
                (self.compact_theta, self.compact_phi),
            ]
            shape_seeds.extend(self._slot_shape_seeds(pose))
            shape_seeds.extend(self._shape_candidates(prev_shape, curr_shape, 1.0)[:8])
        seen = set()
        candidates = []
        arm_entry_margin = self.required_arm_margin
        if terminal_bridge:
            arm_entry_margin = max(0.0, self.required_arm_margin - 0.006)
        for theta, phi in shape_seeds:
            phi = self._normalize_angle_deg(phi)
            key = (round(float(theta), 2), round(float(phi), 2))
            if key in seen:
                continue
            seen.add(key)
            arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(pose, theta, phi)
            if arm_margin < arm_entry_margin:
                continue
            desired_shape = mid_shape
            candidate = self._make_sequence_candidate(
                pose,
                (theta, phi),
                pose,
                desired_shape,
                body_margin,
                body_obstacle,
                body_point,
                arm_margin,
                arm_obstacle,
                arm_point,
                1.0,
                passage_context,
            )
            if candidate is None:
                continue
            continuity = abs(theta - mid_shape[0]) + 0.25 * abs(self._angle_delta(mid_shape[1], phi))
            clearance = min(body_margin, arm_margin)
            candidate["node_cost"] += 0.02 * continuity - 1.0 * min(clearance, self.sequence_risk_arm_margin)
            if terminal_bridge and arm_margin < self.required_arm_margin:
                candidate["node_cost"] += 4.0 * (self.required_arm_margin - arm_margin)
            candidates.append(candidate)
        candidates.sort(key=lambda item: item["node_cost"])
        limit = min(self.edge_repair_candidates_per_knot, 3) if low_release_passage else self.edge_repair_candidates_per_knot
        return candidates[:limit]

    def _edge_repair_swept_feasible(self, candidates):
        poses = [cand["pose"] for cand in candidates]
        shapes = [cand["shape"] for cand in candidates]
        swept = self._swept_validation(poses, shapes)
        return bool(swept.get("swept_feasible", True)), swept

    def _update_terminal_bridge_debug(
        self,
        debug,
        status,
        body_margin=None,
        body_obstacle=None,
        body_point=None,
        arm_margin=None,
        arm_obstacle=None,
        arm_point=None,
    ):
        if debug is None:
            return
        score = min(
            float(body_margin if body_margin is not None else debug.get("terminal_bridge_best_body_margin", float("-inf"))),
            float(arm_margin if arm_margin is not None else debug.get("terminal_bridge_best_arm_margin", float("-inf"))),
        )
        old_score = min(
            float(debug.get("terminal_bridge_best_body_margin", float("-inf"))),
            float(debug.get("terminal_bridge_best_arm_margin", float("-inf"))),
        )
        if score <= old_score:
            return
        debug["terminal_bridge_best_status"] = status
        if body_margin is not None:
            debug["terminal_bridge_best_body_margin"] = float(body_margin)
            debug["terminal_bridge_best_body_obstacle"] = body_obstacle or "none"
            debug["terminal_bridge_best_body_point"] = body_point or (0.0, 0.0, 0.0)
        if arm_margin is not None:
            debug["terminal_bridge_best_arm_margin"] = float(arm_margin)
            debug["terminal_bridge_best_arm_obstacle"] = arm_obstacle or "none"
            debug["terminal_bridge_best_arm_point"] = arm_point or (0.0, 0.0, 0.0)

    def _edge_repair_bridge_for_pair(self, prev_state, curr_cand, ladders, deadline, debug=None):
        prev_cand = prev_state["candidate"]
        best = None
        for ladder in ladders:
            if time.monotonic() >= deadline:
                return best, "timeout"
            stages = [
                {
                    "cost": prev_state["cost"],
                    "candidate": prev_cand,
                    "prev_candidate": prev_state.get("prev_candidate"),
                    "bridge_candidates": [],
                }
            ]
            for knot_idx, pose in enumerate(ladder):
                ratio = float(knot_idx + 1) / float(len(ladder) + 1)
                knot_candidates = self._edge_repair_candidates_for_pose(
                    pose,
                    prev_cand,
                    curr_cand,
                    ratio,
                    terminal_bridge=debug is not None,
                )
                if not knot_candidates:
                    body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
                    arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(
                        pose,
                        self.compact_theta,
                        self.compact_phi,
                        sample_limit=7,
                    )
                    self._update_terminal_bridge_debug(
                        debug,
                        "knot_candidate_empty",
                        body_margin,
                        body_obstacle,
                        body_point,
                        arm_margin,
                        arm_obstacle,
                        arm_point,
                    )
                    stages = []
                    break
                next_stage = []
                for state in stages:
                    for knot_cand in knot_candidates:
                        if time.monotonic() >= deadline:
                            return best, "timeout"
                        edge = self._sequence_transition_cost(
                            state["candidate"],
                            knot_cand,
                            state.get("prev_candidate"),
                            include_safety=True,
                        )
                        if edge is None:
                            detail = self._sequence_edge_required_detail(state["candidate"], knot_cand)
                            self._update_terminal_bridge_debug(
                                debug,
                                "knot_edge_unsafe",
                                detail["min_body_margin"],
                                detail["min_body_obstacle"],
                                detail["min_body_point"],
                                detail["min_arm_margin"],
                                detail["min_arm_obstacle"],
                                detail["min_arm_point"],
                            )
                            continue
                        total = state["cost"] + edge + knot_cand["node_cost"]
                        next_stage.append(
                            {
                                "cost": total,
                                "candidate": knot_cand,
                                "prev_candidate": state["candidate"],
                                "bridge_candidates": state["bridge_candidates"] + [knot_cand],
                            }
                        )
                if not next_stage:
                    stages = []
                    break
                next_stage.sort(key=lambda item: item["cost"])
                stages = next_stage[: self.edge_repair_candidates_per_knot]
            for state in stages:
                edge = self._sequence_transition_cost(
                    state["candidate"],
                    curr_cand,
                    state.get("prev_candidate"),
                    include_safety=True,
                )
                if edge is None:
                    detail = self._sequence_edge_required_detail(state["candidate"], curr_cand)
                    self._update_terminal_bridge_debug(
                        debug,
                        "final_edge_unsafe",
                        detail["min_body_margin"],
                        detail["min_body_obstacle"],
                        detail["min_body_point"],
                        detail["min_arm_margin"],
                        detail["min_arm_obstacle"],
                        detail["min_arm_point"],
                    )
                    continue
                local_sequence = [prev_cand] + state["bridge_candidates"] + [curr_cand]
                feasible, swept = self._edge_repair_swept_feasible(local_sequence)
                if not feasible:
                    self._update_terminal_bridge_debug(
                        debug,
                        "swept_validation_failed",
                        float(swept.get("min_swept_body_margin", float("-inf"))),
                        swept.get("min_swept_body_obstacle", "none"),
                        swept.get("min_swept_body_point", (0.0, 0.0, 0.0)),
                        float(swept.get("min_swept_arm_margin", float("-inf"))),
                        swept.get("min_swept_arm_obstacle", "none"),
                        swept.get("min_swept_arm_point", (0.0, 0.0, 0.0)),
                    )
                    continue
                min_body = float(swept.get("min_swept_body_margin", float("inf")))
                min_arm = float(swept.get("min_swept_arm_margin", float("inf")))
                total = state["cost"] + edge + curr_cand["node_cost"]
                repair_penalty = 0.8 * len(state["bridge_candidates"]) - 1.5 * min(min_body, self.sequence_risk_body_margin) - 1.5 * min(min_arm, self.sequence_risk_arm_margin)
                total += repair_penalty
                item = {
                    "cost": total,
                    "prev": -1,
                    "candidate": curr_cand,
                    "prev_candidate": state["candidate"],
                    "bridge_candidates": state["bridge_candidates"],
                    "edge_repair_swept": swept,
                }
                if best is None or item["cost"] < best["cost"]:
                    best = item
        return best, None

    def _build_edge_repair_layer(self, idx, prev_layer, curr_candidates, info, deadline):
        if not self.edge_repair_enabled or not prev_layer or not curr_candidates:
            return [], None
        if self.edge_repair_time_budget > 0.0:
            deadline = min(deadline, time.monotonic() + self.edge_repair_time_budget)
        pair_queue = []
        for prev_idx, prev_state in prev_layer:
            prev_cand = prev_state["candidate"]
            prev_prev = prev_state.get("prev_candidate")
            for curr_cand in curr_candidates:
                cheap_edge = self._sequence_transition_cost(prev_cand, curr_cand, prev_prev, include_safety=False)
                pair_queue.append((prev_state["cost"] + cheap_edge + curr_cand["node_cost"], prev_idx, prev_state, curr_cand))
        pair_queue.sort(key=lambda item: item[0])
        layer = []
        pair_limit = self.edge_repair_max_pairs
        if len(curr_candidates) > self.sequence_dp_candidates_per_wp:
            pair_limit = max(pair_limit, self.edge_repair_max_pairs * 2)
        for _, prev_idx, prev_state, curr_cand in pair_queue[:pair_limit]:
            if time.monotonic() >= deadline:
                return layer, "timeout"
            info["edge_repair_attempts"] += 1
            terminal_bridge = bool(curr_cand.get("terminal_standoff_lock", False))
            if terminal_bridge:
                info["terminal_bridge_attempts"] += 1
            if curr_cand.get("terminal_standoff_lock", False):
                ladders = self._terminal_edge_repair_pose_ladders(prev_state["candidate"], curr_cand)
                ladders.extend(self._edge_repair_pose_ladders(prev_state["candidate"]["pose"], curr_cand["pose"]))
            else:
                ladders = self._edge_repair_pose_ladders(prev_state["candidate"]["pose"], curr_cand["pose"])
            repaired, status = self._edge_repair_bridge_for_pair(
                prev_state,
                curr_cand,
                ladders,
                deadline,
                info if terminal_bridge else None,
            )
            if status == "timeout":
                return layer, "timeout"
            if repaired is None:
                continue
            repaired["prev"] = prev_idx
            if terminal_bridge:
                info["terminal_bridge_successes"] += 1
                info["terminal_bridge_knots"] = max(
                    info["terminal_bridge_knots"],
                    len(repaired.get("bridge_candidates", [])),
                )
                info["terminal_bridge_best_status"] = "success"
            swept = repaired.get("edge_repair_swept", {})
            info["edge_repair_successes"] += 1
            info["edge_repair_used"] = True
            info["edge_repair_min_body_margin"] = min(
                info["edge_repair_min_body_margin"],
                float(swept.get("min_swept_body_margin", float("inf"))),
            )
            info["edge_repair_min_arm_margin"] = min(
                info["edge_repair_min_arm_margin"],
                float(swept.get("min_swept_arm_margin", float("inf"))),
            )
            layer.append(repaired)
            if len(layer) >= self.edge_repair_max_results:
                break
        layer.sort(key=lambda item: item["cost"])
        return layer, None

    def _refine_whole_body_transit_sequence(self, ego_poses, goal_theta, goal_phi, expanded=False, expanded_indices=None):
        expanded_indices = set(expanded_indices or [])
        repair_mode = bool(expanded or expanded_indices)
        info = self._new_refinement_info(ego_poses)
        info["shape_policy"] = "dp_graph_dynamics_aware"
        info["sequence_optimizer"] = "dp_graph"
        info["expanded_candidates"] = bool(repair_mode)
        info["expanded_candidate_indices"] = len(expanded_indices) if expanded_indices else (len(ego_poses) if expanded else 0)
        if not self.enable_whole_body_refinement or not ego_poses:
            shapes, status = self._assign_transit_shapes(ego_poses)
            return ego_poses, shapes, info, status
        start_time = time.monotonic()
        if self.refine_wp_time_budget > 0.0:
            total_budget = max(self.refine_time_budget, self.refine_wp_time_budget * float(len(ego_poses)))
        else:
            total_budget = self.refine_time_budget
        info["refine_time_budget"] = total_budget
        deadline = start_time + total_budget if total_budget > 0.0 else float("inf")
        if math.isfinite(deadline):
            dp_reserve = min(total_budget * (1.0 - self.sequence_candidate_budget_fraction), self.sequence_dp_min_time_budget)
            dp_reserve = max(dp_reserve, min(self.sequence_dp_min_time_budget, total_budget * 0.25))
            if repair_mode:
                dp_reserve = max(dp_reserve, min(10.0, total_budget * 0.35))
            candidate_stage_budget = max(0.2, total_budget - dp_reserve)
            candidate_deadline = start_time + candidate_stage_budget
        else:
            dp_reserve = 0.0
            candidate_stage_budget = float("inf")
            candidate_deadline = float("inf")
        info["sequence_candidate_budget_sec"] = candidate_stage_budget
        info["sequence_dp_reserved_sec"] = dp_reserve
        passage_schedule, passage_schedule_info = self._build_passage_shape_schedule(ego_poses, goal_theta, goal_phi)
        info.update(passage_schedule_info)
        candidates_by_idx = []
        failures = []
        candidate_start_time = time.monotonic()
        for idx in range(len(ego_poses)):
            self._publish_planning_state("sequence_candidates_wp_%d" % idx)
            if time.monotonic() >= deadline:
                failure = self._candidate_failure_template(idx, ego_poses[idx])
                self._record_refinement_failure(info, failure)
                info["timeout_sec"] = time.monotonic() - start_time
                return None, None, info, "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_%d" % idx
            remaining_wps = max(1, len(ego_poses) - idx)
            if math.isfinite(candidate_deadline):
                remaining_candidate_time = max(0.05, candidate_deadline - time.monotonic())
                dynamic_wp_budget = max(0.12, 1.25 * remaining_candidate_time / float(remaining_wps))
                idx_expanded = bool(expanded or idx in expanded_indices)
                configured_wp_budget = self.refine_wp_time_budget * (2.0 if idx_expanded else 1.0)
                wp_budget = min(configured_wp_budget, dynamic_wp_budget) if configured_wp_budget > 0.0 else dynamic_wp_budget
            else:
                idx_expanded = bool(expanded or idx in expanded_indices)
                wp_budget = self.refine_wp_time_budget * (2.0 if idx_expanded else 1.0)
            wp_deadline = min(deadline, time.monotonic() + wp_budget) if wp_budget > 0.0 else deadline
            candidates, failure = self._generate_sequence_candidates(
                ego_poses,
                idx,
                goal_theta,
                goal_phi,
                idx_expanded,
                wp_deadline,
                passage_schedule,
            )
            candidates_by_idx.append(candidates)
            failures.append(failure)
            info["candidate_count"] += len(candidates)
            info["failure_pose_candidates"] += failure["pose_candidates"]
            info["failure_body_valid_candidates"] += failure["body_valid_candidates"]
            info["failure_shape_candidates"] += failure["shape_candidates"]
            info["failure_arm_valid_candidates"] += failure["arm_valid_candidates"]
            info["passage_risk_wp_count"] += 1 if failure.get("passage_risk", False) else 0
            info["passage_compact_safe_wp_count"] += 1 if failure.get("passage_compact_safe", False) else 0
            info["passage_compact_lock_wp_count"] += 1 if failure.get("passage_compact_lock", False) else 0
            release_factor = float(failure.get("release_factor", 0.0))
            info["release_min"] = min(float(info.get("release_min", float("inf"))), release_factor)
            info["release_max"] = max(float(info.get("release_max", 0.0)), release_factor)
            info["release_mean"] += release_factor
            if not candidates:
                self._record_refinement_failure(info, failure)
                if failure.get("failure_reason") == "timeout":
                    info["timeout_sec"] = time.monotonic() - start_time
                    return None, None, info, "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_%d" % idx
                return None, None, info, "WHOLE_BODY_REFINEMENT_FAILED_WP_%d" % idx
        info["sequence_candidate_generation_sec"] = time.monotonic() - candidate_start_time
        info["release_mean"] = float(info.get("release_mean", 0.0)) / float(max(len(ego_poses), 1))
        if not math.isfinite(float(info.get("release_min", float("inf")))):
            info["release_min"] = 0.0
        candidates_for_dp = []
        dp_max_candidates = 0
        rich_risk_layers = 0
        for idx, candidates in enumerate(candidates_by_idx):
            dp_limit = self.sequence_dp_candidates_per_wp
            idx_expanded = bool(expanded or idx in expanded_indices)
            near_expanded_window = bool(
                expanded
                or idx in expanded_indices
                or (idx - 1) in expanded_indices
                or (idx + 1) in expanded_indices
            )
            pose_diversity = self._candidate_pose_diversity(candidates)
            low_release_passage_layer = any(
                bool(candidate.get("passage_risk", False))
                and float(candidate.get("release_factor", 0.0)) < 0.15
                for candidate in candidates
            )
            approach_release_layer = any(candidate.get("approach_release", False) for candidate in candidates)
            rich_risk_layer = (
                repair_mode
                and near_expanded_window
                and not low_release_passage_layer
                and pose_diversity >= 4
                and len(candidates) > self.sequence_dp_candidates_per_wp * 2
            )
            if rich_risk_layer:
                rich_risk_layers += 1
            if low_release_passage_layer and idx != len(candidates_by_idx) - 1:
                dp_limit = min(dp_limit, max(self.sequence_min_candidates_per_wp, 6))
            elif approach_release_layer:
                dp_limit = max(dp_limit, max(self.sequence_min_candidates_per_wp * 6, 24))
            elif idx_expanded:
                dp_limit = max(dp_limit, max(self.sequence_min_candidates_per_wp * 4, 16))
            elif rich_risk_layer:
                dp_limit = max(dp_limit, max(self.sequence_min_candidates_per_wp * 4, 16))
            if idx == len(candidates_by_idx) - 1:
                # The final standoff layer is a hard connection point before
                # reach-in. Do not discard safe terminal morphologies only
                # because their node cost is slightly worse in isolation.
                dp_limit = max(dp_limit, max(self.sequence_min_candidates_per_wp * 6, 24))
            dp_limit = min(len(candidates), dp_limit)
            dp_max_candidates = max(dp_max_candidates, dp_limit)
            candidates_for_dp.append(
                self._select_candidate_subset(
                    candidates,
                    dp_limit,
                    (idx_expanded or rich_risk_layer or idx == len(candidates_by_idx) - 1) and not low_release_passage_layer,
                )
            )
        info["sequence_dp_candidate_count"] = sum(len(candidates) for candidates in candidates_for_dp)
        info["sequence_dp_max_candidates_per_wp"] = dp_max_candidates
        info["sequence_dp_rich_risk_layers"] = rich_risk_layers
        info["sequence_approach_release_candidate_count"] = sum(
            1 for candidates in candidates_for_dp for candidate in candidates if candidate.get("approach_release", False)
        )
        info["sequence_approach_bridge_candidate_count"] = sum(
            1 for candidates in candidates_for_dp for candidate in candidates if self._is_approach_bridge_candidate(candidate)
        )
        if candidates_for_dp:
            pose_diversities = [self._candidate_pose_diversity(candidates) for candidates in candidates_for_dp]
            info["sequence_dp_min_pose_diversity"] = min(pose_diversities)
            info["sequence_dp_max_pose_diversity"] = max(pose_diversities)

        layers = []
        first = []
        dp_start_time = time.monotonic()
        effective_beam_width = self.sequence_beam_width
        effective_transition_neighbors = self.sequence_transition_neighbors
        effective_edge_eval_neighbors = self.sequence_edge_eval_neighbors
        edge_safety_in_dp = bool(repair_mode)
        if repair_mode:
            effective_beam_width = max(effective_beam_width, 24)
            effective_transition_neighbors = max(effective_transition_neighbors, effective_beam_width)
            effective_edge_eval_neighbors = max(effective_edge_eval_neighbors, 2)
        info["sequence_effective_beam_width"] = effective_beam_width
        info["sequence_effective_transition_neighbors"] = effective_transition_neighbors
        info["sequence_edge_safety_in_dp"] = edge_safety_in_dp
        for cand in candidates_for_dp[0]:
            first.append({"cost": cand["node_cost"], "prev": -1, "candidate": cand, "prev_candidate": None})
        layers.append(self._select_state_subset(first, effective_beam_width, repair_mode))
        for idx in range(1, len(candidates_for_dp)):
            self._publish_planning_state("sequence_dp_wp_%d" % idx)
            if time.monotonic() >= deadline:
                failure = failures[idx]
                failure["failure_reason"] = "timeout"
                self._record_refinement_failure(info, failure)
                info["sequence_dp_sec"] = time.monotonic() - dp_start_time
                info["timeout_sec"] = time.monotonic() - start_time
                return None, None, info, "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_%d" % idx
            layer = []
            prev_layer = self._select_indexed_state_subset(
                list(enumerate(layers[-1])),
                min(effective_transition_neighbors, effective_beam_width),
                repair_mode,
            )
            ordered_prev_by_candidate = []
            for cand in candidates_for_dp[idx]:
                cheap_prev = []
                for prev_idx, prev_state in prev_layer:
                    if time.monotonic() >= deadline:
                        break
                    prev_cand = prev_state["candidate"]
                    prev_prev = prev_state.get("prev_candidate")
                    cheap_edge = self._sequence_transition_cost(prev_cand, cand, prev_prev, include_safety=False)
                    cheap_prev.append((prev_state["cost"] + cheap_edge, prev_idx, prev_state, cheap_edge))
                if not cheap_prev:
                    continue
                cheap_prev.sort(key=lambda item: item[0])
                ordered_prev_by_candidate.append((cand, cheap_prev))

            def build_layer(eval_all_edges):
                built = []
                for cand, cheap_prev in ordered_prev_by_candidate:
                    best_cost = float("inf")
                    best_prev = -1
                    best_prev_candidate = None
                    prev_candidates = cheap_prev if eval_all_edges else cheap_prev[:effective_edge_eval_neighbors]
                    for _, prev_idx, prev_state, _ in prev_candidates:
                        if time.monotonic() >= deadline:
                            return built, "timeout"
                        prev_cand = prev_state["candidate"]
                        prev_prev = prev_state.get("prev_candidate")
                        edge = self._sequence_transition_cost(prev_cand, cand, prev_prev, include_safety=edge_safety_in_dp)
                        info["expanded_edges"] += 1
                        if edge is None:
                            continue
                        total = prev_state["cost"] + edge + cand["node_cost"]
                        if total < best_cost:
                            best_cost = total
                            best_prev = prev_idx
                            best_prev_candidate = prev_cand
                    if best_prev >= 0:
                        built.append({"cost": best_cost, "prev": best_prev, "candidate": cand, "prev_candidate": best_prev_candidate})
                return built, None

            layer, build_status = build_layer(False)
            if build_status == "timeout":
                info["sequence_dp_partial_timeout"] = True
                info["sequence_dp_partial_timeout_wp"] = idx
                if not layer:
                    failure = failures[idx]
                    failure["failure_reason"] = "timeout"
                    self._record_refinement_failure(info, failure)
                    info["sequence_dp_sec"] = time.monotonic() - dp_start_time
                    info["timeout_sec"] = time.monotonic() - start_time
                    return None, None, info, "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_%d" % idx
            if not layer and effective_edge_eval_neighbors < len(prev_layer):
                info["sequence_edge_exhaustive_retries"] += 1
                layer, build_status = build_layer(True)
                if build_status == "timeout":
                    info["sequence_dp_partial_timeout"] = True
                    info["sequence_dp_partial_timeout_wp"] = idx
                    if not layer:
                        failure = failures[idx]
                        failure["failure_reason"] = "timeout"
                        self._record_refinement_failure(info, failure)
                        info["sequence_dp_sec"] = time.monotonic() - dp_start_time
                        info["timeout_sec"] = time.monotonic() - start_time
                        return None, None, info, "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_%d" % idx
            if not layer:
                layer, build_status = self._build_edge_repair_layer(idx, prev_layer, candidates_for_dp[idx], info, deadline)
                if build_status == "timeout":
                    info["edge_repair_timeout"] = True
                    if not layer:
                        failure = failures[idx]
                        failure["failure_reason"] = "edge_repair_timeout"
                        self._record_refinement_failure(info, failure)
                        info["sequence_dp_sec"] = time.monotonic() - dp_start_time
                        return None, None, info, "WHOLE_BODY_SEQUENCE_TRANSITION_FAILED_WP_%d" % idx
            if not layer:
                failure = failures[idx]
                failure["failure_reason"] = "transition_limit"
                self._annotate_transition_failure(failure, prev_layer, candidates_for_dp[idx])
                self._record_refinement_failure(info, failure)
                info["sequence_dp_sec"] = time.monotonic() - dp_start_time
                return None, None, info, "WHOLE_BODY_SEQUENCE_TRANSITION_FAILED_WP_%d" % idx
            if len(layer) > effective_beam_width:
                layer = self._select_state_subset(layer, effective_beam_width, repair_mode)
            else:
                layer.sort(key=lambda item: item["cost"])
            layers.append(layer)
        info["sequence_dp_sec"] = time.monotonic() - dp_start_time

        last_idx, last_state = min(enumerate(layers[-1]), key=lambda item: item[1]["cost"])
        selected_states = []
        cursor = last_idx
        for layer_idx in range(len(layers) - 1, -1, -1):
            state = layers[layer_idx][cursor]
            selected_states.append(state)
            cursor = state["prev"]
            if cursor < 0 and layer_idx > 0:
                cursor = 0
        selected_states.reverse()
        selected = []
        selected_bridge_knots = 0
        for state_idx, state in enumerate(selected_states):
            if state_idx > 0:
                bridge_candidates = state.get("bridge_candidates", [])
                selected_bridge_knots += len(bridge_candidates)
                selected.extend(bridge_candidates)
            selected.append(state["candidate"])
        if selected_bridge_knots > 0:
            info["edge_repair_used"] = True
            info["edge_repair_inserted_knots"] = selected_bridge_knots

        refined_poses = [c["pose"] for c in selected]
        refined_shapes = [c["shape"] for c in selected]
        total_dev = 0.0
        total_com = 0.0
        total_tau = 0.0
        morphology_activation_sum = 0.0
        morphology_activated = 0
        for idx, cand in enumerate(selected):
            total_dev += cand["ego_dev"]
            total_com += cand["coupling"]["com_offset"]
            total_tau += cand["coupling"]["gravity_torque"]
            morphology_activation_sum += cand["morphology_activation"]
            if cand["morphology_activation"] > 0.5:
                morphology_activated += 1
            if cand.get("passage_risk", False):
                info["selected_passage_risk_count"] += 1
            if cand.get("passage_compact_lock", False):
                info["selected_passage_compact_lock_count"] += 1
            desired = self._desired_transit_shape(idx, len(ego_poses), goal_theta, goal_phi)
            if cand["ego_dev"] > 1e-3:
                info["pose_refined_count"] += 1
            if abs(cand["shape"][0] - desired[0]) > 1e-3 or abs(self._angle_delta(desired[1], cand["shape"][1])) > 1e-3:
                info["arm_optimized_count"] += 1
            info["max_ego_deviation"] = max(info["max_ego_deviation"], cand["ego_dev"])
            info["max_arm_com_offset"] = max(info["max_arm_com_offset"], cand["coupling"]["com_offset"])
            info["max_gravity_torque"] = max(info["max_gravity_torque"], cand["coupling"]["gravity_torque"])
            info["min_candidate_body_margin"] = min(info["min_candidate_body_margin"], cand["body_margin"])
            info["min_candidate_arm_margin"] = min(info["min_candidate_arm_margin"], cand["arm_margin"])
        count = float(max(len(selected), 1))
        info["whole_body_refinement_used"] = True
        info["sequence_cost"] = last_state["cost"]
        info["mean_ego_deviation"] = total_dev / count
        info["mean_arm_com_offset"] = total_com / count
        info["mean_gravity_torque"] = total_tau / count
        info["morphology_activated_wp_count"] = morphology_activated
        info["mean_morphology_activation"] = morphology_activation_sum / count
        return refined_poses, refined_shapes, info, "OK"

    def _refine_whole_body_transit(self, ego_poses, goal_theta, goal_phi):
        info = {
            "whole_body_refinement_used": False,
            "refinement_input_points": len(ego_poses),
            "pose_refined_count": 0,
            "arm_optimized_count": 0,
            "max_ego_deviation": 0.0,
            "mean_ego_deviation": 0.0,
            "max_arm_com_offset": 0.0,
            "mean_arm_com_offset": 0.0,
            "max_arm_lateral_extent": 0.0,
            "transit_compact_weight": self.refine_compact_weight,
            "transit_com_weight": self.refine_com_weight,
            "transit_clearance_weight": self.refine_clearance_weight,
            "refine_min_arm_margin": self.refine_min_arm_margin,
            "morphology_gate": "home_clearance_activated",
            "morphology_home_hold_weight": self.morphology_home_hold_weight,
            "morphology_activation_margin": self.morphology_activation_margin,
            "morphology_activation_band": self.morphology_activation_band,
            "morphology_min_active_theta": self.morphology_min_active_theta,
            "morphology_activated_wp_count": 0,
            "mean_morphology_activation": 0.0,
            "valid_candidate_target": self.refine_valid_candidate_target,
            "shape_policy": "adaptive_dense_grid_first_feasible" if self.refine_first_feasible else "adaptive_dense_grid_anytime_best",
            "refine_first_feasible": self.refine_first_feasible,
            "refine_time_budget": self.refine_time_budget,
            "refine_wp_time_budget": self.refine_wp_time_budget,
            "shape_candidates_per_pose": self.refine_shape_candidates_per_pose,
            "timeout_sec": 0.0,
            "failure_wp": -1,
            "failure_pose": (0.0, 0.0, 0.0),
            "failure_body_best_margin": float("-inf"),
            "failure_body_obstacle": "none",
            "failure_body_point": (0.0, 0.0, 0.0),
            "failure_arm_best_margin": float("-inf"),
            "failure_arm_obstacle": "none",
            "failure_arm_point": (0.0, 0.0, 0.0),
            "failure_pose_candidates": 0,
            "failure_body_valid_candidates": 0,
            "failure_shape_candidates": 0,
            "failure_arm_valid_candidates": 0,
        }
        if not self.enable_whole_body_refinement or not ego_poses:
            shapes, status = self._assign_transit_shapes(ego_poses)
            return ego_poses, shapes, info, status

        refined_poses = []
        refined_shapes = []
        total_deviation = 0.0
        lateral_offsets = [0.0, -self.refine_lateral_step, self.refine_lateral_step]
        z_offsets = [
            0.0,
            -0.5 * self.refine_z_step,
            0.5 * self.refine_z_step,
            self.refine_z_step,
            1.5 * self.refine_z_step,
            2.0 * self.refine_z_step,
            2.5 * self.refine_z_step,
            3.0 * self.refine_z_step,
            4.0 * self.refine_z_step,
            5.0 * self.refine_z_step,
        ]
        yaw_offsets = [0.0]
        previous_shape = (self.current_theta_phi[0], self.current_theta_phi[1])
        previous_pose = None
        total_com_offset = 0.0
        total_morphology_activation = 0.0
        morphology_activated_count = 0
        start_time = time.monotonic()
        if self.refine_wp_time_budget > 0.0:
            total_budget = max(self.refine_time_budget, self.refine_wp_time_budget * float(len(ego_poses)))
        else:
            total_budget = self.refine_time_budget
        deadline = start_time + total_budget if total_budget > 0.0 else float("inf")
        info["refine_time_budget"] = total_budget

        for idx, ego_pose in enumerate(ego_poses):
            self._publish_planning_state("refine_transit_wp_%d" % idx)
            wp_start_time = time.monotonic()
            wp_deadline = min(deadline, wp_start_time + self.refine_wp_time_budget) if self.refine_wp_time_budget > 0.0 else deadline
            path_yaw = self._path_yaw_at(ego_poses, idx)
            desired_shape = self._desired_transit_shape(idx, len(ego_poses), goal_theta, goal_phi)
            best = None
            best_cost = float("inf")
            failure = {
                "wp": idx,
                "pose": self._pose_xyz(ego_pose),
                "body_best_margin": float("-inf"),
                "body_obstacle": "none",
                "body_point": (0.0, 0.0, 0.0),
                "arm_best_margin": float("-inf"),
                "arm_obstacle": "none",
                "arm_point": (0.0, 0.0, 0.0),
                "pose_candidates": 0,
                "body_valid_candidates": 0,
                "shape_candidates": 0,
                "arm_valid_candidates": 0,
            }
            if time.monotonic() >= wp_deadline:
                self._record_refinement_failure(info, failure)
                info["timeout_sec"] = time.monotonic() - start_time
                return None, None, info, "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_%d" % idx
            search_sets = [
                (lateral_offsets, z_offsets, yaw_offsets),
                (
                    [-2.0 * self.refine_lateral_step, -self.refine_lateral_step, 0.0, self.refine_lateral_step, 2.0 * self.refine_lateral_step],
                    [
                        -self.refine_z_step,
                        -0.5 * self.refine_z_step,
                        0.0,
                        0.5 * self.refine_z_step,
                        self.refine_z_step,
                        1.5 * self.refine_z_step,
                        2.0 * self.refine_z_step,
                        2.5 * self.refine_z_step,
                        3.0 * self.refine_z_step,
                        3.5 * self.refine_z_step,
                        4.0 * self.refine_z_step,
                        4.5 * self.refine_z_step,
                        5.0 * self.refine_z_step,
                    ],
                    [-self.refine_yaw_step_deg, 0.0, self.refine_yaw_step_deg],
                ),
            ]
            for local_lateral_offsets, local_z_offsets, local_yaw_offsets_base in search_sets:
                pose_offsets = [(0.0, 0.0)] if idx == len(ego_poses) - 1 else self._ordered_pose_offsets(local_lateral_offsets, local_z_offsets)
                local_yaw_offsets = [0.0] if idx == len(ego_poses) - 1 else self._ordered_yaw_offsets(local_yaw_offsets_base)
                for lateral, dz in pose_offsets:
                    for yaw_offset in local_yaw_offsets:
                        pose = self._copy_pose(ego_pose) if idx == len(ego_poses) - 1 else self._offset_pose(ego_pose, path_yaw, lateral, dz, yaw_offset)
                        failure["pose_candidates"] += 1
                        body_margin, body_obstacle, body_point = self._body_point_margin(self._pose_xyz(pose))
                        if body_margin > failure["body_best_margin"]:
                            failure["body_best_margin"] = body_margin
                            failure["body_obstacle"] = body_obstacle
                            failure["body_point"] = body_point
                        if body_margin < 0.0:
                            continue
                        failure["body_valid_candidates"] += 1
                        home_margin = self._arm_margin(pose, self.compact_theta, self.compact_phi, sample_limit=5)
                        morphology_activation = self._morphology_activation(home_margin, False)
                        shape_candidates = self._shape_candidates(previous_shape, desired_shape, morphology_activation)
                        for shape_rank, (theta, phi) in enumerate(shape_candidates):
                            if shape_rank >= self.refine_shape_candidates_per_pose:
                                break
                            if (
                                morphology_activation > 0.5
                                and idx != len(ego_poses) - 1
                                and abs(theta - self.compact_theta) < self.morphology_min_active_theta
                            ):
                                continue
                            if time.monotonic() >= wp_deadline:
                                if best is not None:
                                    break
                                self._record_refinement_failure(info, failure)
                                info["timeout_sec"] = time.monotonic() - start_time
                                return None, None, info, "WHOLE_BODY_REFINEMENT_TIMEOUT_WP_%d" % idx
                            failure["shape_candidates"] += 1
                            fast_arm_margin, fast_arm_obstacle, fast_arm_point = self._arm_margin_detail(pose, theta, phi, sample_limit=5)
                            if fast_arm_margin < self.refine_min_arm_margin:
                                if fast_arm_margin > failure["arm_best_margin"]:
                                    failure["arm_best_margin"] = fast_arm_margin
                                    failure["arm_obstacle"] = fast_arm_obstacle
                                    failure["arm_point"] = fast_arm_point
                                continue
                            arm_margin, arm_obstacle, arm_point = self._arm_margin_detail(pose, theta, phi)
                            if arm_margin > failure["arm_best_margin"]:
                                failure["arm_best_margin"] = arm_margin
                                failure["arm_obstacle"] = arm_obstacle
                                failure["arm_point"] = arm_point
                            if arm_margin < self.refine_min_arm_margin:
                                continue
                            failure["arm_valid_candidates"] += 1
                            ego_dev = point_distance(self._pose_xyz(ego_pose), self._pose_xyz(pose))
                            smooth = 0.0 if previous_pose is None else point_distance(self._pose_xyz(previous_pose), self._pose_xyz(pose))
                            theta_step = abs(theta - previous_shape[0])
                            phi_step = abs(self._angle_delta(previous_shape[1], phi))
                            desired_cost = abs(theta - desired_shape[0]) + 0.25 * abs(self._angle_delta(phi, desired_shape[1]))
                            compact_cost = abs(theta - self.compact_theta) + 0.25 * abs(self._angle_delta(phi, self.compact_phi))
                            com_offset, lateral_extent = self._arm_morphology_metrics(pose, theta, phi)
                            clearance_reward = min(body_margin, self.refine_clearance_target) + min(arm_margin, self.refine_clearance_target)
                            min_clearance = min(body_margin, arm_margin)
                            clearance_deficit = max(0.0, self.refine_clearance_target - min_clearance) / max(self.refine_clearance_target, 1e-3)
                            arm_clearance_deficit = max(0.0, self.refine_clearance_target - arm_margin) / max(self.refine_clearance_target, 1e-3)
                            home_hold_cost = (
                                (1.0 - morphology_activation)
                                * self.morphology_home_hold_weight
                                * compact_cost
                            )
                            transit_morphology_cost = (
                                self.refine_compact_weight * compact_cost
                                + self.refine_com_weight * com_offset
                                + self.refine_lateral_extent_weight * lateral_extent
                                + home_hold_cost
                            )
                            cost = (
                                4.0 * ego_dev
                                + 0.08 * smooth
                                + 0.025 * theta_step
                                + 0.008 * phi_step
                                + 0.01 * desired_cost
                                + transit_morphology_cost
                                + self.refine_clearance_weight * clearance_deficit * clearance_deficit
                                + 0.5 * self.refine_clearance_weight * arm_clearance_deficit * arm_clearance_deficit
                                - 1.2 * clearance_reward
                            )
                            if cost < best_cost:
                                best_cost = cost
                                best = (pose, (theta, phi), ego_dev, com_offset, lateral_extent, morphology_activation)
                                if self.refine_first_feasible:
                                    break
                            if (not self.refine_first_feasible) and failure["arm_valid_candidates"] >= self.refine_valid_candidate_target:
                                break
                        if best is not None and self.refine_first_feasible:
                            break
                        if best is not None and time.monotonic() >= wp_deadline:
                            break
                        if (not self.refine_first_feasible) and failure["arm_valid_candidates"] >= self.refine_valid_candidate_target:
                            break
                    if best is not None and self.refine_first_feasible:
                        break
                    if best is not None and time.monotonic() >= wp_deadline:
                        break
                    if (not self.refine_first_feasible) and failure["arm_valid_candidates"] >= self.refine_valid_candidate_target:
                        break
                if best is not None:
                    break
            if best is None:
                self._record_refinement_failure(info, failure)
                return None, None, info, "WHOLE_BODY_REFINEMENT_FAILED_WP_%d" % idx
            pose, shape, ego_dev, com_offset, lateral_extent, morphology_activation = best
            if ego_dev > 1e-3:
                info["pose_refined_count"] += 1
            if abs(shape[0] - desired_shape[0]) > 1e-3 or abs(self._angle_delta(desired_shape[1], shape[1])) > 1e-3:
                info["arm_optimized_count"] += 1
            refined_poses.append(pose)
            refined_shapes.append(shape)
            previous_pose = pose
            previous_shape = shape
            total_deviation += ego_dev
            total_com_offset += com_offset
            total_morphology_activation += morphology_activation
            if morphology_activation > 0.5:
                morphology_activated_count += 1
            info["max_ego_deviation"] = max(info["max_ego_deviation"], ego_dev)
            info["max_arm_com_offset"] = max(info["max_arm_com_offset"], com_offset)
            info["max_arm_lateral_extent"] = max(info["max_arm_lateral_extent"], lateral_extent)

        info["whole_body_refinement_used"] = True
        info["mean_ego_deviation"] = total_deviation / float(max(len(refined_poses), 1))
        info["mean_arm_com_offset"] = total_com_offset / float(max(len(refined_poses), 1))
        info["morphology_activated_wp_count"] = morphology_activated_count
        info["mean_morphology_activation"] = total_morphology_activation / float(max(len(refined_poses), 1))
        return refined_poses, refined_shapes, info, "OK"

    def _record_refinement_failure(self, info, failure):
        info["failure_wp"] = failure["wp"]
        info["failure_pose"] = failure["pose"]
        info["failure_body_best_margin"] = failure["body_best_margin"]
        info["failure_body_obstacle"] = failure["body_obstacle"]
        info["failure_body_point"] = failure["body_point"]
        info["failure_arm_best_margin"] = failure["arm_best_margin"]
        info["failure_arm_obstacle"] = failure["arm_obstacle"]
        info["failure_arm_point"] = failure["arm_point"]
        info["failure_pose_candidates"] = failure["pose_candidates"]
        info["failure_body_valid_candidates"] = failure["body_valid_candidates"]
        info["failure_shape_candidates"] = failure["shape_candidates"]
        info["failure_arm_valid_candidates"] = failure["arm_valid_candidates"]
        info["failure_reason"] = failure.get("failure_reason", "none")
        info["failure_passage_risk"] = bool(failure.get("passage_risk", False))
        info["failure_passage_compact_lock"] = bool(failure.get("passage_compact_lock", False))
        info["failure_passage_compact_safe"] = bool(failure.get("passage_compact_safe", False))
        info["failure_passage_slot"] = failure.get("passage_slot", "none")
        info["failure_release_factor"] = float(failure.get("release_factor", 0.0))
        info["release_at_failure"] = float(failure.get("release_factor", 0.0))
        info["failure_approach_release"] = bool(failure.get("approach_release", False))
        info["failure_stable_shape"] = failure.get("stable_shape", (0.0, 0.0))
        info["failure_q_ref"] = failure.get("q_ref", (0.0, 0.0))
        hold_shape = failure.get("passage_hold_shape")
        info["failure_passage_hold_used"] = hold_shape is not None
        info["failure_passage_hold_shape"] = hold_shape or (0.0, 0.0)
        info["failure_passage_hold_segment_id"] = int(failure.get("passage_hold_segment_id", -1))
        info["transition_prev_candidate_count"] = int(failure.get("transition_prev_candidate_count", 0))
        info["transition_curr_candidate_count"] = int(failure.get("transition_curr_candidate_count", 0))
        info["transition_prev_pose"] = failure.get("transition_prev_pose", (0.0, 0.0, 0.0))
        info["transition_curr_pose"] = failure.get("transition_curr_pose", (0.0, 0.0, 0.0))
        info["transition_prev_shape"] = failure.get("transition_prev_shape", (0.0, 0.0))
        info["transition_curr_shape"] = failure.get("transition_curr_shape", (0.0, 0.0))
        info["transition_edge_dist"] = float(failure.get("transition_edge_dist", 0.0))
        info["transition_edge_dtheta"] = float(failure.get("transition_edge_dtheta", 0.0))
        info["transition_edge_dphi"] = float(failure.get("transition_edge_dphi", 0.0))
        info["transition_edge_steps"] = int(failure.get("transition_edge_steps", 0))
        info["transition_min_body_margin"] = float(failure.get("transition_min_body_margin", float("inf")))
        info["transition_min_body_obstacle"] = failure.get("transition_min_body_obstacle", "none")
        info["transition_min_body_point"] = failure.get("transition_min_body_point", (0.0, 0.0, 0.0))
        info["transition_min_arm_margin"] = float(failure.get("transition_min_arm_margin", float("inf")))
        info["transition_min_arm_obstacle"] = failure.get("transition_min_arm_obstacle", "none")
        info["transition_min_arm_point"] = failure.get("transition_min_arm_point", (0.0, 0.0, 0.0))

    def _angle_delta(self, value, goal):
        return math.degrees(math.atan2(math.sin(math.radians(goal - value)), math.cos(math.radians(goal - value))))

    def _arm_key(self, theta, phi):
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)
        return (int(round(theta / theta_step)), int(round(phi / phi_step)))

    def _arm_value(self, key):
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)
        return (key[0] * theta_step, key[1] * phi_step)

    def _arm_heuristic(self, key, goal_key):
        theta, phi = self._arm_value(key)
        goal_theta, goal_phi = self._arm_value(goal_key)
        return abs(theta - goal_theta) + 0.25 * abs(self._angle_delta(phi, goal_phi))

    def _search_reach_in(self, standoff_pose, start_theta, start_phi, goal_theta, goal_phi):
        start_time = time.monotonic()
        deadline = start_time + self.reach_search_time_budget if self.reach_search_time_budget > 0.0 else float("inf")
        self.last_reach_search_info = {
            "reach_search_time_budget": self.reach_search_time_budget,
            "reach_search_timeout_sec": 0.0,
            "reach_search_expanded": 0,
            "reach_search_max_states": self.reach_search_max_states,
        }
        start_key = self._arm_key(start_theta, start_phi)
        goal_key = self._arm_key(goal_theta, goal_phi)
        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        phi_min = float(self.arm_params["phi_min_deg"])
        phi_max = float(self.arm_params["phi_max_deg"])
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)

        def valid_key(key):
            if time.monotonic() >= deadline:
                return False
            theta, phi = self._arm_value(key)
            if theta < theta_min - 1e-9 or theta > theta_max + 1e-9:
                return False
            if phi < phi_min - 1e-9 or phi > phi_max + 1e-9:
                return False
            margin, _, _ = self._arm_margin_detail(standoff_pose, theta, phi)
            return margin >= self.required_arm_margin

        def valid_edge(a_key, b_key):
            if time.monotonic() >= deadline:
                return False
            a_theta, a_phi = self._arm_value(a_key)
            b_theta, b_phi = self._arm_value(b_key)
            dtheta = abs(b_theta - a_theta)
            dphi = abs(self._angle_delta(a_phi, b_phi))
            steps = max(
                2,
                int(math.ceil(dtheta / max(self.swept_theta_step, 1e-3))),
                int(math.ceil(dphi / max(self.swept_phi_step, 1e-3))),
            )
            for step in range(1, steps):
                ratio = float(step) / float(steps)
                theta = a_theta + (b_theta - a_theta) * ratio
                phi = self._normalize_angle_deg(a_phi + self._angle_delta(a_phi, b_phi) * ratio)
                margin, _, _ = self._arm_margin_detail(standoff_pose, theta, phi)
                if margin < self.required_arm_margin:
                    return False
            return True

        if not valid_key(start_key):
            start_key = self._arm_key(self.compact_theta, self.compact_phi)
            if not valid_key(start_key):
                return None
        if not valid_key(goal_key):
            return None

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        open_heap = [(self._arm_heuristic(start_key, goal_key), 0.0, start_key)]
        came_from = {}
        g_score = {start_key: 0.0}
        closed = set()
        while open_heap and len(closed) < self.reach_search_max_states:
            if time.monotonic() >= deadline:
                self.last_reach_search_info["reach_search_timeout_sec"] = time.monotonic() - start_time
                self.last_reach_search_info["reach_search_expanded"] = len(closed)
                return None
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_key:
                keys = [current]
                while keys[-1] != start_key:
                    keys.append(came_from[keys[-1]])
                keys.reverse()
                return [self._arm_value(key) for key in keys]
            closed.add(current)
            self.last_reach_search_info["reach_search_expanded"] = len(closed)
            for dt, dp in neighbors:
                nb = (current[0] + dt, current[1] + dp)
                if nb in closed or not valid_key(nb) or not valid_edge(current, nb):
                    continue
                step = math.sqrt((dt * theta_step) ** 2 + (dp * phi_step * 0.25) ** 2)
                tentative = current_g + step
                if tentative >= g_score.get(nb, float("inf")):
                    continue
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_heap, (tentative + self._arm_heuristic(nb, goal_key), tentative, nb))
        self.last_reach_search_info["reach_search_timeout_sec"] = time.monotonic() - start_time
        self.last_reach_search_info["reach_search_expanded"] = len(closed)
        return None

    def _run_refine_whole_body_transit(self, ego_poses, goal_theta, goal_phi, expanded=False, expanded_indices=None):
        previous = self.cloud_collision_enabled
        self.cloud_collision_enabled = self.static_cloud_in_optimization
        try:
            if self.sequence_optimizer_enabled:
                return self._refine_whole_body_transit_sequence(
                    ego_poses,
                    goal_theta,
                    goal_phi,
                    expanded=expanded,
                    expanded_indices=expanded_indices,
                )
            return self._refine_whole_body_transit(ego_poses, goal_theta, goal_phi)
        finally:
            self.cloud_collision_enabled = previous

    def _run_search_reach_in(self, standoff_pose, start_theta, start_phi, goal_theta, goal_phi):
        previous = self.cloud_collision_enabled
        self.cloud_collision_enabled = self.static_cloud_in_optimization
        try:
            return self._search_reach_in(standoff_pose, start_theta, start_phi, goal_theta, goal_phi)
        finally:
            self.cloud_collision_enabled = previous

    def _run_safety_report(self, poses, transit_shapes):
        previous = self.cloud_collision_enabled
        self.cloud_collision_enabled = self.static_cloud_in_validation
        try:
            return self._safety_report(poses, transit_shapes)
        finally:
            self.cloud_collision_enabled = previous

    def _tip_error(self, pose, theta, phi):
        pts = world_arm_points(pose, theta, phi, self.arm_params)
        return point_distance(pts[-1], tuple(self.target["pose"][:3]))

    def _refresh_latched_inputs(self):
        self.input_refresh_attempts += 1
        self.global_path_param_cache_used = False
        self.last_global_path_wait_ok = self.global_path is not None and bool(self.global_path.poses)
        self.last_ego_reference_wait_ok = self.reference_path is not None and bool(self.reference_path.poses)
        self.reach_candidates_param_cache_used = False
        if self.global_path is None or not self.global_path.poses:
            path = self._global_path_from_param_cache()
            if path is not None and path.poses:
                self.global_path = path
                self.global_path_stamp = rospy.Time.now()
                self.global_path_array_cb_count += 1
                self.last_global_path_wait_ok = True
                self.global_path_param_cache_used = True
        if self.global_path is None or not self.global_path.poses:
            try:
                msg = rospy.wait_for_message("/planner/global_path", Path, timeout=self.input_wait_timeout)
                self._global_path_cb(msg)
                self.last_global_path_wait_ok = bool(msg.poses)
            except rospy.ROSException:
                self.last_global_path_wait_ok = False
        if self.global_path is None or not self.global_path.poses:
            try:
                msg = rospy.wait_for_message("/planner/global_path_array", Float32MultiArray, timeout=self.input_wait_timeout)
                self._global_path_array_cb(msg)
                self.last_global_path_wait_ok = self.global_path is not None and bool(self.global_path.poses)
            except rospy.ROSException:
                self.last_global_path_wait_ok = False
        if self.reference_path is None or not self.reference_path.poses:
            path = self._reference_from_param_cache()
            if path is not None and path.poses:
                self.reference_path = path
                self.reference_array_cb_count += 1
                self.last_ego_reference_wait_ok = True
                self.ego_reference_param_cache_used = True
        if self.reference_path is None or not self.reference_path.poses:
            try:
                msg = rospy.wait_for_message("/planner/ego_reference_trajectory", Path, timeout=self.input_wait_timeout)
                self._reference_cb(msg)
                self.last_ego_reference_wait_ok = bool(msg.poses)
            except rospy.ROSException:
                self.last_ego_reference_wait_ok = False
        if self.reference_path is None or not self.reference_path.poses:
            try:
                msg = rospy.wait_for_message("/planner/ego_reference_trajectory_array", Float32MultiArray, timeout=self.input_wait_timeout)
                self._reference_array_cb(msg)
                self.last_ego_reference_wait_ok = self.reference_path is not None and bool(self.reference_path.poses)
            except rospy.ROSException:
                self.last_ego_reference_wait_ok = False
        self.last_reach_candidates_wait_ok = bool(self.reach_candidates)
        if not self.reach_candidates:
            candidates = self._reach_candidates_from_param_cache()
            if candidates:
                self.last_reach_candidates_wait_ok = True
                self.reach_candidates_param_cache_used = True
        if not self.reach_candidates:
            try:
                msg = rospy.wait_for_message("/planner/reach_candidates", Float32MultiArray, timeout=self.input_wait_timeout)
                self._reach_candidates_cb(msg)
                self.last_reach_candidates_wait_ok = bool(self.reach_candidates)
            except rospy.ROSException:
                self.last_reach_candidates_wait_ok = False

    def _publish_planning_state(self, stage="start"):
        self.plan_pub.publish(
            String(
                data=(
                    "status=PLANNING_WHOLE_BODY\n"
                    "planner_version=%s\n"
                    "method=ours_full_min\n"
                    "scenario=%s\n"
                    "planning_stage=%s\n"
                    "global_path_received=%s\n"
                    "global_path_points=%d\n"
                    "global_path_age=%s\n"
                    "ego_reference_received=%s\n"
                    "ego_reference_points=%d\n"
                    "reach_candidate_count=%d\n"
                    "static_cloud_in_optimization=%s\n"
                    "static_cloud_in_validation=%s\n"
                    "static_cloud_loaded=%s\n"
                    "static_cloud_esdf=%s\n"
                    "static_cloud_query_padding=%.3f\n"
                    "refine_time_budget=%.2f\n"
                    "refine_wp_time_budget=%.2f\n"
                    "refine_first_feasible=%s\n"
                    "refine_valid_candidate_target=%d\n"
                    "shape_candidates_per_pose=%d\n"
                    "sequence_min_candidates_per_wp=%d\n"
                    "refine_min_arm_margin=%.3f\n"
                    "sequence_optimizer_enabled=%s\n"
                    "sequence_beam_width=%d\n"
                    "sequence_edge_eval_neighbors=%d\n"
                    "reference_keyframes_after_compression=%d\n"
                    "reference_keyframes_pruned=%d\n"
                    "reference_edge_splits=%d\n"
                    "reference_edge_detours=%d\n"
                    "reference_edge_unresolved=%d\n"
                    "independent_skeleton_used=%s\n"
                    "independent_skeleton_points=%d\n"
                    "robust_margin_enabled=%s\n"
                    "swept_validation_enabled=%s\n"
                    "required_body_margin=%.3f\n"
                    "required_arm_margin=%.3f\n"
                    "m_uav=%.3f\n"
                    "m_arm_total=%.3f\n"
                    "hard_com_limit=%.3f\n"
                    "tau_limit=%.3f\n"
                    "v_max=%.3f\n"
                    "a_max=%.3f\n"
                    "j_max=%.3f\n"
                    "tilt_max_deg=%.2f\n"
                    "morphology_gate=home_clearance_activated\n"
                    "morphology_home_hold_weight=%.3f\n"
                    "morphology_activation_margin=%.3f\n"
                    "morphology_min_active_theta=%.3f\n"
                    "passage_morphology_gate_enabled=%s\n"
                    "passage_compact_shape_weight=%.3f\n"
                    "passage_transition_shape_weight=%.3f\n"
                    "morphology_release_start_fraction=%.3f\n"
                    "morphology_release_target_distance=%.3f\n"
                    "reach_search_time_budget=%.2f\n"
                    "reach_search_max_states=%d\n"
                    "enable_reference_safety_repair=%s\n"
                    "enable_global_repair_fallback=%s"
                )
                % (
                    PLANNER_VERSION,
                    self.scenario_name,
                    stage,
                    str(self.global_path is not None),
                    len(self.global_path.poses) if self.global_path is not None else 0,
                    self._fmt(self._global_path_age()),
                    str(self.reference_path is not None),
                    len(self.reference_path.poses) if self.reference_path is not None else 0,
                    len(self.reach_candidates),
                    str(self.static_cloud_in_optimization),
                    str(self.static_cloud_in_validation),
                    str(self.static_cloud_loaded),
                    getattr(self.static_occupancy, "distance_field_status", "unknown"),
                    self.static_cloud_query_padding,
                    self.refine_time_budget,
                    self.refine_wp_time_budget,
                    str(self.refine_first_feasible),
                    self.refine_valid_candidate_target,
                    self.refine_shape_candidates_per_pose,
                    self.sequence_min_candidates_per_wp,
                    self.refine_min_arm_margin,
                    str(self.sequence_optimizer_enabled),
                    self.sequence_beam_width,
                    self.sequence_edge_eval_neighbors,
                    int(self.last_reference_repair_info.get("reference_keyframes_after_compression", 0)),
                    int(self.last_reference_repair_info.get("reference_keyframes_pruned", 0)),
                    int(self.last_reference_repair_info.get("reference_edge_splits", 0)),
                    int(self.last_reference_repair_info.get("reference_edge_detours", 0)),
                    int(self.last_reference_repair_info.get("reference_edge_unresolved", 0)),
                    str(self.last_reference_repair_info.get("independent_skeleton_used", False)),
                    int(self.last_reference_repair_info.get("independent_skeleton_points", 0)),
                    str(self.robust_margin_enabled),
                    str(self.swept_validation_enabled),
                    self.required_body_margin,
                    self.required_arm_margin,
                    self.m_uav,
                    self.m_arm_total,
                    self.hard_com_limit,
                    self.tau_limit,
                    self.v_max,
                    self.a_max,
                    self.j_max,
                    math.degrees(self.tilt_max),
                    self.morphology_home_hold_weight,
                    self.morphology_activation_margin,
                    self.morphology_min_active_theta,
                    str(self.passage_morphology_gate_enabled),
                    self.passage_compact_shape_weight,
                    self.passage_transition_shape_weight,
                    self.morphology_release_start_fraction,
                    self.morphology_release_target_distance,
                    self.reach_search_time_budget,
                    self.reach_search_max_states,
                    str(self.enable_reference_safety_repair),
                    str(self.enable_global_repair_fallback),
                )
            )
        )

    def _finalize_refined_plan(self, poses, transit_shapes, candidate, selected_idx, reference_repair, refinement):
        poses = [self._copy_pose(pose) for pose in poses]
        transit_shapes = list(transit_shapes)
        standoff_pose = candidate["pose"]
        goal_theta, goal_phi = candidate["theta_phi"]
        start_theta, start_phi = transit_shapes[-1]
        self._publish_planning_state("reach_in")
        reach_shapes = self._run_search_reach_in(standoff_pose, start_theta, start_phi, goal_theta, goal_phi)
        if reach_shapes is None:
            self.last_failure_info = refinement
            self.last_failure_info.update(self.last_reach_search_info)
            return None, "REACH_IN_BACKBONE_COLLISION", {"poses": poses, "arm_shapes": transit_shapes}

        for theta, phi in reach_shapes[1:]:
            poses.append(self._copy_pose(standoff_pose))
            transit_shapes.append((theta, phi))

        final_tip_error = self._tip_error(standoff_pose, transit_shapes[-1][0], transit_shapes[-1][1])
        if final_tip_error > self.target_tolerance:
            self.last_failure_info = refinement
            return None, "TIP_ERROR_TOO_HIGH", {"poses": poses, "arm_shapes": transit_shapes}

        self._publish_planning_state("interpolate")
        poses, transit_shapes, interpolation = self._interpolate_arm_limited(poses, transit_shapes)

        self._publish_planning_state("final_static_cloud_validation")
        safety = self._run_safety_report(poses, transit_shapes)
        dynamics = self._refine_time_allocation(poses, transit_shapes)
        swept = self._swept_validation(poses, transit_shapes)
        safety.update(dynamics)
        safety.update(swept)
        attempt = {"poses": poses, "arm_shapes": transit_shapes, "safety": safety, "interpolation": interpolation}
        if safety["body_margin"] < self.required_body_margin:
            self.last_failure_info = self._validation_failure_info(safety, refinement)
            return None, "BODY_COLLISION_VALIDATION", attempt
        if safety["arm_margin"] < self.required_arm_margin:
            self.last_failure_info = self._validation_failure_info(safety, refinement)
            return None, "ARM_BACKBONE_COLLISION_VALIDATION", attempt
        if not safety.get("swept_feasible", True):
            self.last_failure_info = self._validation_failure_info(safety, refinement)
            return None, "SWEPT_COLLISION_VALIDATION", attempt
        if not safety.get("dynamics_feasible", True):
            self.last_failure_info = self._validation_failure_info(safety, refinement)
            return None, "UAV_DYNAMICS_VALIDATION", attempt
        if not safety.get("arm_motion_feasible", True):
            self.last_failure_info = self._validation_failure_info(safety, refinement)
            return None, "ARM_MOTION_VALIDATION", attempt

        return {
            "poses": poses,
            "arm_shapes": transit_shapes,
            "candidate": candidate,
            "selected_idx": selected_idx,
            "tip_error": final_tip_error,
            "safety": safety,
            "reference_repair": reference_repair,
            "refinement": refinement,
            "interpolation": interpolation,
        }, "WHOLE_BODY_PLAN_READY", attempt

    def _build_plan(self):
        self.last_failure_info = {}
        self._clear_planning_caches()
        if self.current_pose is None:
            return None, "WAITING_CURRENT_POSE"
        needs_static_cloud = self.use_static_cloud and (
            self.require_static_cloud
            or self.static_cloud_in_optimization
            or self.static_cloud_in_validation
        )
        if needs_static_cloud and (not self.static_cloud_loaded or not self.static_occupancy.ready()):
            return None, "WAITING_STATIC_CLOUD"
        self._publish_runtime_status("REFRESH_INPUTS_BEGIN")
        self._refresh_latched_inputs()
        self._publish_runtime_status(
            "REFRESH_INPUTS_DONE",
            global_path_wait_ok=str(self.last_global_path_wait_ok),
            ego_reference_wait_ok=str(self.last_ego_reference_wait_ok),
            reach_candidates_wait_ok=str(self.last_reach_candidates_wait_ok),
            global_path_param_cache_used=str(self.global_path_param_cache_used),
            ego_reference_param_cache_used=str(self.ego_reference_param_cache_used),
            reach_candidates_param_cache_used=str(self.reach_candidates_param_cache_used),
        )
        has_global_path = self.global_path is not None and bool(self.global_path.poses)
        has_ego_reference = self.reference_path is not None and bool(self.reference_path.poses)
        if not has_ego_reference:
            return None, "WAITING_EGO_REFERENCE"
        if not self.reach_candidates:
            return None, "WAITING_REACH_CANDIDATES"

        raw_poses = self._path_poses(self.reference_path)
        approach_reference_poses = None
        if raw_poses:
            self._publish_runtime_status("REFERENCE_WINDOW_BEGIN", raw_reference_points=len(raw_poses))
            compressed_for_selection = self._local_refinement_window(raw_poses)
            if compressed_for_selection:
                approach_reference_poses = compressed_for_selection
            self._publish_runtime_status(
                "REFERENCE_WINDOW_DONE",
                raw_reference_points=len(raw_poses),
                approach_reference_points=len(approach_reference_poses) if approach_reference_poses is not None else 0,
            )
        self._publish_runtime_status("SELECT_REACH_CANDIDATE_BEGIN")
        candidate, selected_idx = self._find_candidate_for_selected_goal(approach_reference_poses)
        if candidate is None:
            return None, "NO_REACH_CANDIDATE"
        self._publish_runtime_status("SELECT_REACH_CANDIDATE_DONE", selected_candidate=selected_idx)
        self.last_selected_candidate_detail = candidate.get("selection_detail", {})
        self._publish_planning_state("prepare_reference")

        reference_repair = {
            "ego_reference_points": len(raw_poses),
            "unsafe_ego_points": 0,
            "global_repair_points": 0,
            "global_repair_used": False,
            "global_repair_reason": "disabled_wait_for_ego",
            "first_unsafe_ego_idx": -1,
            "first_unsafe_ego_margin": float("inf"),
            "first_unsafe_ego_obstacle": "none",
            "first_unsafe_ego_pose": (0.0, 0.0, 0.0),
            "first_unsafe_ego_point": (0.0, 0.0, 0.0),
        }
        base_poses = list(raw_poses)
        standoff_pose = candidate["pose"]
        poses, reference_repair, slot_repair = self._prepare_transit_reference(base_poses, standoff_pose, reference_repair)
        reference_repair["reference_edges_unresolved_before_dp"] = self._reference_edges_unresolved(reference_repair)
        self.last_reference_repair_info = dict(reference_repair)

        goal_theta, goal_phi = candidate["theta_phi"]
        self._publish_planning_state("refine_transit")
        transit_input_poses = list(poses)
        reference_edges_unsafe = (
            int(reference_repair.get("reference_edge_unresolved", 0)) > 0
            and (
                float(reference_repair.get("reference_edge_min_body_margin", float("inf"))) < self.required_body_margin
                or float(reference_repair.get("reference_edge_min_arm_margin", float("inf"))) < self.required_arm_margin
            )
        )
        poses, transit_shapes, refinement, status = self._run_refine_whole_body_transit(poses, goal_theta, goal_phi)
        refinement.update(slot_repair)
        if reference_edges_unsafe:
            refinement["reference_edges_deferred_to_swept_validation"] = True
            refinement["fallback_reason"] = "reference_edges_deferred_to_swept_validation"
        if transit_shapes is None:
            raw_refinement_status = status
            if (
                self.local_repair_enabled
                and self.sequence_optimizer_enabled
                and self._transition_failure_wp(status) is not None
            ):
                repair_input_poses = list(transit_input_poses)
                total_inserted = 0
                repair_status = status
                attempts = max(1, self.local_repair_max_attempts)
                visited_repair_windows = set()
                for attempt in range(attempts):
                    inserted = 0
                    failure_idx = self._transition_failure_wp(repair_status)
                    expanded_indices = set()
                    if failure_idx is not None:
                        repair_input_poses, inserted = self._insert_transition_repair_waypoints(repair_input_poses, repair_status)
                        total_inserted += inserted
                        expanded_indices = self._transition_repair_window_indices(len(repair_input_poses), failure_idx, inserted)
                        window_key = (min(expanded_indices), max(expanded_indices)) if expanded_indices else (-1, -1)
                        if window_key in visited_repair_windows and inserted <= 0:
                            break
                        visited_repair_windows.add(window_key)
                    elif attempt > 0:
                        break
                    self._publish_planning_state("local_repair_refine_transit_%d" % (attempt + 1))
                    poses, transit_shapes, refinement, status = self._run_refine_whole_body_transit(
                        repair_input_poses,
                        goal_theta,
                        goal_phi,
                        expanded_indices=expanded_indices,
                    )
                    refinement.update(slot_repair)
                    refinement["fallback_reason"] = raw_refinement_status
                    refinement["local_repair_used"] = True
                    refinement["local_repair_window_start"] = min(expanded_indices) if expanded_indices else -1
                    refinement["local_repair_window_end"] = max(expanded_indices) if expanded_indices else -1
                    refinement["transition_repair_inserted"] = total_inserted
                    refinement["local_repair_attempts"] = attempt + 1
                    if transit_shapes is not None:
                        break
                    repair_status = status
                    if inserted <= 0 and self._transition_failure_wp(repair_status) is None:
                        break
                if transit_shapes is None:
                    refinement["local_repair_used"] = True
            if transit_shapes is not None:
                pass
            elif str(refinement.get("failure_reason", "")).startswith("terminal_standoff_lock_"):
                refinement["fallback_reason"] = "disabled_after_%s" % refinement.get("failure_reason", "terminal_standoff_lock")
                self.last_failure_info = refinement
                return None, status
            elif refinement.get("failure_passage_slot", "none") != "none":
                self._publish_planning_state("slot_corridor_fallback")
                corridor_reference = dict(reference_repair)
                corridor_poses, corridor_info = self._build_slot_skeleton(standoff_pose)
                corridor_reference.update(corridor_info)
                corridor_poses, corridor_reference, corridor_slot_repair = self._prepare_transit_reference(
                    corridor_poses,
                    standoff_pose,
                    corridor_reference,
                )
                self.last_reference_repair_info = dict(corridor_reference)
                poses, transit_shapes, corridor_refinement, corridor_status = self._run_refine_whole_body_transit(
                    corridor_poses,
                    goal_theta,
                    goal_phi,
                )
                corridor_refinement.update(corridor_slot_repair)
                corridor_refinement["fallback_reason"] = "slot_corridor_after_%s" % raw_refinement_status
                corridor_refinement["slot_corridor_fallback_used"] = True
                corridor_refinement["slot_corridor_points"] = len(corridor_poses)
                corridor_refinement["transition_repair_inserted"] = int(refinement.get("transition_repair_inserted", 0))
                corridor_refinement["swept_repair_inserted"] = int(refinement.get("swept_repair_inserted", 0))
                if transit_shapes is None:
                    refinement = corridor_refinement
                    reference_repair = corridor_reference
                    status = corridor_status
                    if not self.enable_global_repair_fallback:
                        self.last_failure_info = refinement
                        return None, status
                else:
                    refinement = corridor_refinement
                    reference_repair = corridor_reference
            elif not self.enable_global_repair_fallback:
                if transit_shapes is None:
                    refinement["fallback_reason"] = "disabled_after_%s" % raw_refinement_status
                    self.last_failure_info = refinement
                    return None, status
            else:
                refinement["fallback_reason"] = "body_astar_disabled_after_%s" % raw_refinement_status
                self.last_failure_info = refinement
                return None, status
        else:
            refinement["fallback_reason"] = "none"

        result, final_status, final_attempt = self._finalize_refined_plan(poses, transit_shapes, candidate, selected_idx, reference_repair, refinement)
        if result is not None:
            return result, final_status

        if final_status == "SWEPT_COLLISION_VALIDATION" and self.local_repair_enabled and self.sequence_optimizer_enabled:
            repair_input_poses = list(transit_input_poses)
            total_inserted = int(refinement.get("transition_repair_inserted", 0))
            attempts = max(1, self.local_repair_max_attempts)
            repair_status = final_status
            for attempt in range(attempts):
                transition_failure_idx = self._transition_failure_wp(repair_status)
                if transition_failure_idx is not None:
                    repair_input_poses, inserted = self._insert_transition_repair_waypoints(
                        repair_input_poses,
                        repair_status,
                    )
                    expanded_indices = self._transition_repair_window_indices(
                        len(repair_input_poses),
                        transition_failure_idx,
                        inserted,
                    )
                else:
                    swept_segment = int(final_attempt.get("safety", {}).get("swept_failure_segment", -1))
                    repair_input_poses, inserted, mapped_failure_idx = self._insert_swept_repair_waypoints(
                        repair_input_poses,
                        final_attempt.get("poses", []),
                        final_attempt.get("arm_shapes", []),
                        swept_segment,
                    )
                    expanded_indices = self._transition_repair_window_indices(
                        len(repair_input_poses),
                        mapped_failure_idx,
                        inserted,
                    )
                if inserted <= 0:
                    break
                total_inserted += inserted
                self._publish_planning_state("swept_repair_refine_transit_%d" % (attempt + 1))
                repair_poses, repair_shapes, repair_refinement, repair_status = self._run_refine_whole_body_transit(
                    repair_input_poses,
                    goal_theta,
                    goal_phi,
                    expanded_indices=expanded_indices,
                )
                repair_refinement["fallback_reason"] = final_status
                repair_refinement.update(slot_repair)
                repair_refinement["local_repair_used"] = True
                repair_refinement["local_repair_window_start"] = min(expanded_indices) if expanded_indices else -1
                repair_refinement["local_repair_window_end"] = max(expanded_indices) if expanded_indices else -1
                repair_refinement["transition_repair_inserted"] = total_inserted
                repair_refinement["local_repair_attempts"] = attempt + 1
                repair_refinement["swept_repair_inserted"] = total_inserted
                if repair_shapes is None:
                    refinement = repair_refinement
                    self.last_failure_info = repair_refinement
                    final_status = repair_status
                    continue
                result, final_status, final_attempt = self._finalize_refined_plan(
                    repair_poses,
                    repair_shapes,
                    candidate,
                    selected_idx,
                    reference_repair,
                    repair_refinement,
                )
                if result is not None:
                    return result, final_status
                if final_status != "SWEPT_COLLISION_VALIDATION":
                    return None, final_status

        return None, final_status

    def _safety_report(self, poses, arm_shapes):
        body = {"margin": float("inf"), "obstacle": "none", "point": (0.0, 0.0, 0.0), "idx": -1, "pose": (0.0, 0.0, 0.0)}
        arm = {"margin": float("inf"), "obstacle": "none", "point": (0.0, 0.0, 0.0), "idx": -1, "pose": (0.0, 0.0, 0.0)}
        arm_required = self.arm_params["radius"] + self.arm_clearance
        arm_sample_count = 0
        arm_below_target_count = 0
        arm_below_min_count = 0
        max_com_offset = 0.0
        mean_com_offset = 0.0
        max_gravity_torque = 0.0
        mean_gravity_torque = 0.0
        for idx, (pose, shape) in enumerate(zip(poses, arm_shapes)):
            point = self._pose_xyz(pose)
            margin, name, near = self._body_point_margin(point)
            if margin < body["margin"]:
                body = {"margin": margin, "obstacle": name, "point": near, "idx": idx, "pose": point}
            coupling = self._coupling_metrics(pose, shape[0], shape[1])
            max_com_offset = max(max_com_offset, coupling["com_offset"])
            mean_com_offset += coupling["com_offset"]
            max_gravity_torque = max(max_gravity_torque, coupling["gravity_torque"])
            mean_gravity_torque += coupling["gravity_torque"]
            for arm_point in world_arm_points(pose, shape[0], shape[1], self.arm_params):
                margin, name, near = self._point_margin(arm_point, arm_required, False)
                arm_sample_count += 1
                if margin < self.refine_clearance_target:
                    arm_below_target_count += 1
                if margin < self.required_arm_margin:
                    arm_below_min_count += 1
                if margin < arm["margin"]:
                    arm = {"margin": margin, "obstacle": name, "point": near, "idx": idx, "pose": point}
        count = float(max(len(poses), 1))
        mean_com_offset /= count
        mean_gravity_torque /= count
        max_theta_step = 0.0
        max_phi_step = 0.0
        theta_tv = 0.0
        phi_tv = 0.0
        for a, b in zip(arm_shapes, arm_shapes[1:]):
            dt = abs(b[0] - a[0])
            dp = abs(self._angle_delta(a[1], b[1]))
            theta_tv += dt
            phi_tv += dp
            max_theta_step = max(max_theta_step, dt)
            max_phi_step = max(max_phi_step, dp)
        return {
            "body_margin": body["margin"],
            "body_obstacle": body["obstacle"],
            "body_point": body["point"],
            "body_idx": body["idx"],
            "body_pose": body["pose"],
            "arm_margin": arm["margin"],
            "arm_obstacle": arm["obstacle"],
            "arm_point": arm["point"],
            "arm_idx": arm["idx"],
            "arm_pose": arm["pose"],
            "arm_sample_count": arm_sample_count,
            "arm_below_target_count": arm_below_target_count,
            "arm_below_min_count": arm_below_min_count,
            "max_theta_step": max_theta_step,
            "max_phi_step": max_phi_step,
            "theta_total_variation": theta_tv,
            "phi_total_variation": phi_tv,
            "static_cloud_ready": self.static_occupancy.ready(),
            "static_cloud_voxels": len(self.static_occupancy.occupied),
            "required_body_margin": self.required_body_margin,
            "required_arm_margin": self.required_arm_margin,
            "max_system_com_offset": max_com_offset,
            "mean_system_com_offset": mean_com_offset,
            "max_gravity_torque": max_gravity_torque,
            "mean_gravity_torque": mean_gravity_torque,
            "com_limit": self.hard_com_limit,
            "tau_limit": self.tau_limit,
        }

    def _validation_failure_info(self, safety, refinement):
        info = dict(refinement) if refinement else {}
        body_margin = float(safety.get("body_margin", float("inf")))
        arm_margin = float(safety.get("arm_margin", float("inf")))
        if arm_margin <= body_margin:
            failure_idx = int(safety.get("arm_idx", -1))
            failure_pose = safety.get("arm_pose", (0.0, 0.0, 0.0))
        else:
            failure_idx = int(safety.get("body_idx", -1))
            failure_pose = safety.get("body_pose", (0.0, 0.0, 0.0))
        info.update(
            {
                "failure_wp": failure_idx,
                "failure_pose": failure_pose,
                "failure_body_best_margin": body_margin,
                "failure_body_obstacle": safety.get("body_obstacle", "none"),
                "failure_body_point": safety.get("body_point", (0.0, 0.0, 0.0)),
                "failure_arm_best_margin": arm_margin,
                "failure_arm_obstacle": safety.get("arm_obstacle", "none"),
                "failure_arm_point": safety.get("arm_point", (0.0, 0.0, 0.0)),
                "failure_reason": safety.get("failure_reason", "validation"),
                "swept_failure_segment": int(safety.get("swept_failure_segment", -1)),
                "min_swept_body_margin": safety.get("min_swept_body_margin", float("inf")),
                "min_swept_arm_margin": safety.get("min_swept_arm_margin", float("inf")),
                "max_system_com_offset": safety.get("max_system_com_offset", 0.0),
                "max_gravity_torque": safety.get("max_gravity_torque", 0.0),
                "max_velocity": safety.get("max_velocity", 0.0),
                "max_acceleration": safety.get("max_acceleration", 0.0),
                "max_jerk": safety.get("max_jerk", 0.0),
                "max_thrust": safety.get("max_thrust", 0.0),
                "max_tilt": safety.get("max_tilt", 0.0),
                "max_theta_dot": safety.get("max_theta_dot", 0.0),
                "max_phi_dot": safety.get("max_phi_dot", 0.0),
            }
        )
        return info

    def _path_msg(self, poses):
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        for pose in poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = pose
            msg.poses.append(ps)
        return msg

    def _arm_msg(self, shapes):
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="waypoint", size=len(shapes), stride=len(shapes) * 2),
            MultiArrayDimension(label="theta_phi", size=2, stride=2),
        ]
        msg.layout.data_offset = 0
        for theta, phi in shapes:
            msg.data.extend([theta, phi])
        return msg

    def _fmt(self, value):
        if math.isinf(value):
            return "-inf" if value < 0.0 else "inf"
        return "%.3f" % value

    def _global_path_age(self):
        if self.global_path_stamp == rospy.Time(0):
            return float("inf")
        return max(0.0, (rospy.Time.now() - self.global_path_stamp).to_sec())

    def _plan_text(self, status, result):
        if result is None:
            lines = [
                "status=%s" % status,
                "planner_version=%s" % PLANNER_VERSION,
                "method=ours_full_min",
                "scenario=%s" % self.scenario_name,
                "freeze_plan_when_ready=%s" % str(self.freeze_plan_when_ready),
                "frozen_plan=%s" % str(self.frozen_plan_result is not None),
                "global_path_received=%s" % str(self.global_path is not None),
                "global_path_points=%d" % (len(self.global_path.poses) if self.global_path is not None else 0),
                "global_path_age=%s" % self._fmt(self._global_path_age()),
                "global_path_cb_count=%d" % self.global_path_cb_count,
                "global_path_array_cb_count=%d" % self.global_path_array_cb_count,
                "global_path_param_cache_used=%s" % str(self.global_path_param_cache_used),
                "ego_reference_received=%s" % str(self.reference_path is not None),
                "ego_reference_points=%d" % (len(self.reference_path.poses) if self.reference_path is not None else 0),
                "ego_reference_cb_count=%d" % self.reference_cb_count,
                "ego_reference_array_cb_count=%d" % self.reference_array_cb_count,
                "ego_reference_param_cache_used=%s" % str(self.ego_reference_param_cache_used),
                "reach_candidate_count=%d" % len(self.reach_candidates),
                "reach_candidates_param_cache_used=%s" % str(self.reach_candidates_param_cache_used),
                "static_cloud_loaded=%s" % str(self.static_cloud_loaded),
                "static_cloud_esdf=%s" % getattr(self.static_occupancy, "distance_field_status", "unknown"),
                "input_refresh_attempts=%d" % self.input_refresh_attempts,
                "input_wait_timeout=%.2f" % self.input_wait_timeout,
                "global_path_wait_ok=%s" % str(self.last_global_path_wait_ok),
                "ego_reference_wait_ok=%s" % str(self.last_ego_reference_wait_ok),
                "reach_candidates_wait_ok=%s" % str(self.last_reach_candidates_wait_ok),
            ]
            failure = self.last_failure_info
            if failure:
                selection_detail = self.last_selected_candidate_detail or {}
                lines.extend(
                    [
                        "selected_candidate_body_margin=%s obstacle=%s"
                        % (
                            self._fmt(float(selection_detail.get("body_margin", float("inf")))),
                            selection_detail.get("body_obstacle", "none"),
                        ),
                        "selected_candidate_goal_arm_margin=%s obstacle=%s"
                        % (
                            self._fmt(float(selection_detail.get("goal_arm_margin", float("inf")))),
                            selection_detail.get("goal_arm_obstacle", "none"),
                        ),
                        "selected_candidate_reach_swept_arm_margin=%s obstacle=%s"
                        % (
                            self._fmt(float(selection_detail.get("swept_arm_margin", float("inf")))),
                            selection_detail.get("swept_arm_obstacle", "none"),
                        ),
                        "selected_standoff=[%.3f, %.3f, %.3f]"
                        % selection_detail.get("standoff_xyz", (0.0, 0.0, 0.0)),
                        "selected_candidate_strict_safe=%s"
                        % str(selection_detail.get("standoff_strict_safe", False)),
                        "selected_candidate_hard_safe=%s"
                        % str(selection_detail.get("standoff_hard_safe", False)),
                        "strict_reach_candidate_count=%d"
                        % int(selection_detail.get("strict_candidate_count", 0)),
                        "hard_safe_reach_candidate_count=%d"
                        % int(selection_detail.get("hard_safe_candidate_count", 0)),
                        "selected_candidate_approach_body_margin=%s"
                        % self._fmt(float(selection_detail.get("approach_body_margin", float("inf")))),
                        "selected_candidate_approach_arm_margin=%s"
                        % self._fmt(float(selection_detail.get("approach_arm_margin", float("inf")))),
                        "selected_candidate_approach_hard_unsafe=%s"
                        % str(selection_detail.get("approach_hard_unsafe", False)),
                        "approach_safe_candidate_count=%d"
                        % int(selection_detail.get("approach_safe_candidate_count", 0)),
                        "failure_wp=%d" % int(failure.get("failure_wp", -1)),
                        "failure_pose=[%.3f, %.3f, %.3f]" % failure.get("failure_pose", (0.0, 0.0, 0.0)),
                        "failure_body_best_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
                        % (
                            self._fmt(float(failure.get("failure_body_best_margin", float("-inf")))),
                            failure.get("failure_body_obstacle", "none"),
                            failure.get("failure_body_point", (0.0, 0.0, 0.0))[0],
                            failure.get("failure_body_point", (0.0, 0.0, 0.0))[1],
                            failure.get("failure_body_point", (0.0, 0.0, 0.0))[2],
                        ),
                        "failure_arm_best_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
                        % (
                            self._fmt(float(failure.get("failure_arm_best_margin", float("-inf")))),
                            failure.get("failure_arm_obstacle", "none"),
                            failure.get("failure_arm_point", (0.0, 0.0, 0.0))[0],
                            failure.get("failure_arm_point", (0.0, 0.0, 0.0))[1],
                            failure.get("failure_arm_point", (0.0, 0.0, 0.0))[2],
                        ),
                        "failure_reason=%s" % failure.get("failure_reason", "none"),
                        "failure_passage_risk=%s" % str(failure.get("failure_passage_risk", False)),
                        "failure_passage_compact_lock=%s" % str(failure.get("failure_passage_compact_lock", False)),
                        "failure_passage_compact_safe=%s" % str(failure.get("failure_passage_compact_safe", False)),
                        "failure_passage_slot=%s" % failure.get("failure_passage_slot", "none"),
                        "failure_release_factor=%.3f" % float(failure.get("failure_release_factor", 0.0)),
                        "failure_approach_release=%s" % str(failure.get("failure_approach_release", False)),
                        "failure_stable_shape=[%.2f, %.2f]" % failure.get("failure_stable_shape", (0.0, 0.0)),
                        "failure_q_ref=[%.2f, %.2f]" % failure.get("failure_q_ref", (0.0, 0.0)),
                        "failure_passage_hold_used=%s" % str(failure.get("failure_passage_hold_used", False)),
                        "failure_passage_hold_shape=[%.2f, %.2f]" % failure.get("failure_passage_hold_shape", (0.0, 0.0)),
                        "failure_passage_hold_segment_id=%d" % int(failure.get("failure_passage_hold_segment_id", -1)),
                        "transition_prev_candidate_count=%d" % int(failure.get("transition_prev_candidate_count", 0)),
                        "transition_curr_candidate_count=%d" % int(failure.get("transition_curr_candidate_count", 0)),
                        "transition_prev_pose=[%.3f, %.3f, %.3f]" % failure.get("transition_prev_pose", (0.0, 0.0, 0.0)),
                        "transition_curr_pose=[%.3f, %.3f, %.3f]" % failure.get("transition_curr_pose", (0.0, 0.0, 0.0)),
                        "transition_prev_shape=[%.2f, %.2f]" % failure.get("transition_prev_shape", (0.0, 0.0)),
                        "transition_curr_shape=[%.2f, %.2f]" % failure.get("transition_curr_shape", (0.0, 0.0)),
                        "transition_edge_dist=%.3f" % float(failure.get("transition_edge_dist", 0.0)),
                        "transition_edge_dtheta=%.2f" % float(failure.get("transition_edge_dtheta", 0.0)),
                        "transition_edge_dphi=%.2f" % float(failure.get("transition_edge_dphi", 0.0)),
                        "transition_edge_steps=%d" % int(failure.get("transition_edge_steps", 0)),
                        "transition_min_body_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
                        % (
                            self._fmt(float(failure.get("transition_min_body_margin", float("inf")))),
                            failure.get("transition_min_body_obstacle", "none"),
                            failure.get("transition_min_body_point", (0.0, 0.0, 0.0))[0],
                            failure.get("transition_min_body_point", (0.0, 0.0, 0.0))[1],
                            failure.get("transition_min_body_point", (0.0, 0.0, 0.0))[2],
                        ),
                        "transition_min_arm_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
                        % (
                            self._fmt(float(failure.get("transition_min_arm_margin", float("inf")))),
                            failure.get("transition_min_arm_obstacle", "none"),
                            failure.get("transition_min_arm_point", (0.0, 0.0, 0.0))[0],
                            failure.get("transition_min_arm_point", (0.0, 0.0, 0.0))[1],
                            failure.get("transition_min_arm_point", (0.0, 0.0, 0.0))[2],
                        ),
                        "swept_failure_segment=%d" % int(failure.get("swept_failure_segment", -1)),
                        "min_swept_body_margin=%s" % self._fmt(float(failure.get("min_swept_body_margin", float("inf")))),
                        "min_swept_arm_margin=%s" % self._fmt(float(failure.get("min_swept_arm_margin", float("inf")))),
                        "max_system_com_offset=%.3f" % float(failure.get("max_system_com_offset", 0.0)),
                        "max_gravity_torque=%.3f" % float(failure.get("max_gravity_torque", 0.0)),
                        "max_velocity=%.3f" % float(failure.get("max_velocity", 0.0)),
                        "max_acceleration=%.3f" % float(failure.get("max_acceleration", 0.0)),
                        "max_jerk=%.3f" % float(failure.get("max_jerk", 0.0)),
                        "max_thrust=%.3f" % float(failure.get("max_thrust", 0.0)),
                        "max_tilt_deg=%.2f" % math.degrees(float(failure.get("max_tilt", 0.0))),
                        "max_theta_dot=%.2f" % float(failure.get("max_theta_dot", 0.0)),
                        "max_phi_dot=%.2f" % float(failure.get("max_phi_dot", 0.0)),
                        "failure_pose_candidates=%d" % int(failure.get("failure_pose_candidates", 0)),
                        "failure_body_valid_candidates=%d" % int(failure.get("failure_body_valid_candidates", 0)),
                        "failure_shape_candidates=%d" % int(failure.get("failure_shape_candidates", 0)),
                        "failure_arm_valid_candidates=%d" % int(failure.get("failure_arm_valid_candidates", 0)),
                        "candidate_count=%d" % int(failure.get("candidate_count", 0)),
                        "slot_keyframes_inserted=%d" % int(failure.get("slot_keyframes_inserted", 0)),
                        "slot_poses_projected=%d" % int(failure.get("slot_poses_projected", 0)),
                        "slot_keyframe_constraints=%s" % ",".join(failure.get("slot_keyframe_constraints", [])),
                        "reference_edge_splits=%d" % int(failure.get("reference_edge_splits", 0)),
                        "reference_edge_detours=%d" % int(failure.get("reference_edge_detours", 0)),
                        "reference_edge_unresolved=%d" % int(failure.get("reference_edge_unresolved", 0)),
                        "reference_edge_densify_timeout=%s" % str(failure.get("reference_edge_densify_timeout", False)),
                        "reference_edge_min_body_margin=%s" % self._fmt(float(failure.get("reference_edge_min_body_margin", float("inf")))),
                        "reference_edge_min_arm_margin=%s" % self._fmt(float(failure.get("reference_edge_min_arm_margin", float("inf")))),
                        "reference_keyframes_pruned=%d" % int(failure.get("reference_keyframes_pruned", 0)),
                        "reference_keyframes_after_compression=%d" % int(failure.get("reference_keyframes_after_compression", 0)),
                        "terminal_tail_pruned=%d" % int(failure.get("terminal_tail_pruned", 0)),
                        "independent_skeleton_used=%s" % str(failure.get("independent_skeleton_used", False)),
                        "independent_skeleton_points=%d" % int(failure.get("independent_skeleton_points", 0)),
                        "body_skeleton_used=%s" % str(failure.get("body_skeleton_used", False)),
                        "body_skeleton_status=%s" % failure.get("body_skeleton_status", "none"),
                        "body_skeleton_expanded=%d" % int(failure.get("body_skeleton_expanded", 0)),
                        "body_skeleton_raw_points=%d" % int(failure.get("body_skeleton_raw_points", 0)),
                        "body_skeleton_start_margin=%s" % self._fmt(float(failure.get("body_skeleton_start_margin", float("-inf")))),
                        "body_skeleton_goal_margin=%s" % self._fmt(float(failure.get("body_skeleton_goal_margin", float("-inf")))),
                        "slot_corridor_fallback_used=%s" % str(failure.get("slot_corridor_fallback_used", False)),
                        "slot_corridor_points=%d" % int(failure.get("slot_corridor_points", 0)),
                        "expanded_edges=%d" % int(failure.get("expanded_edges", 0)),
                        "sequence_beam_width=%d" % int(failure.get("sequence_beam_width", self.sequence_beam_width)),
                        "sequence_effective_beam_width=%d" % int(failure.get("sequence_effective_beam_width", self.sequence_beam_width)),
                        "sequence_edge_eval_neighbors=%d" % int(failure.get("sequence_edge_eval_neighbors", self.sequence_edge_eval_neighbors)),
                        "sequence_effective_transition_neighbors=%d" % int(failure.get("sequence_effective_transition_neighbors", self.sequence_transition_neighbors)),
                        "sequence_edge_exhaustive_retries=%d" % int(failure.get("sequence_edge_exhaustive_retries", 0)),
                        "sequence_edge_collision_lazy=%s" % str(failure.get("sequence_edge_collision_lazy", True)),
                        "sequence_edge_safety_in_dp=%s" % str(failure.get("sequence_edge_safety_in_dp", False)),
                        "expanded_candidate_indices=%d" % int(failure.get("expanded_candidate_indices", 0)),
                        "local_repair_window_start=%d" % int(failure.get("local_repair_window_start", -1)),
                        "local_repair_window_end=%d" % int(failure.get("local_repair_window_end", -1)),
                        "sequence_dp_candidates_per_wp=%d" % int(failure.get("sequence_dp_candidates_per_wp", self.sequence_dp_candidates_per_wp)),
                        "sequence_dp_max_candidates_per_wp=%d" % int(failure.get("sequence_dp_max_candidates_per_wp", self.sequence_dp_candidates_per_wp)),
                        "sequence_dp_min_pose_diversity=%d" % int(failure.get("sequence_dp_min_pose_diversity", 0)),
                        "sequence_dp_max_pose_diversity=%d" % int(failure.get("sequence_dp_max_pose_diversity", 0)),
                        "sequence_dp_rich_risk_layers=%d" % int(failure.get("sequence_dp_rich_risk_layers", 0)),
                        "sequence_approach_release_candidate_count=%d" % int(failure.get("sequence_approach_release_candidate_count", 0)),
                        "sequence_approach_bridge_candidate_count=%d" % int(failure.get("sequence_approach_bridge_candidate_count", 0)),
                        "sequence_dp_candidate_count=%d" % int(failure.get("sequence_dp_candidate_count", 0)),
                        "sequence_candidate_budget_sec=%.2f" % float(failure.get("sequence_candidate_budget_sec", 0.0)),
                        "sequence_dp_reserved_sec=%.2f" % float(failure.get("sequence_dp_reserved_sec", 0.0)),
                        "sequence_candidate_generation_sec=%.2f" % float(failure.get("sequence_candidate_generation_sec", 0.0)),
                        "sequence_dp_sec=%.2f" % float(failure.get("sequence_dp_sec", 0.0)),
                        "sequence_dp_partial_timeout=%s" % str(failure.get("sequence_dp_partial_timeout", False)),
                        "sequence_dp_partial_timeout_wp=%d" % int(failure.get("sequence_dp_partial_timeout_wp", -1)),
                        "edge_body_margin_target=%.3f" % float(failure.get("edge_body_margin_target", self.edge_body_margin_target)),
                        "edge_arm_margin_target=%.3f" % float(failure.get("edge_arm_margin_target", self.edge_arm_margin_target)),
                        "sequence_risk_body_margin=%.3f" % float(failure.get("sequence_risk_body_margin", self.sequence_risk_body_margin)),
                        "sequence_risk_arm_margin=%.3f" % float(failure.get("sequence_risk_arm_margin", self.sequence_risk_arm_margin)),
                        "sequence_min_candidates_per_wp=%d" % int(failure.get("sequence_min_candidates_per_wp", self.sequence_min_candidates_per_wp)),
                        "passage_morphology_gate_enabled=%s" % str(failure.get("passage_morphology_gate_enabled", self.passage_morphology_gate_enabled)),
                        "passage_risk_wp_count=%d" % int(failure.get("passage_risk_wp_count", 0)),
                        "passage_compact_safe_wp_count=%d" % int(failure.get("passage_compact_safe_wp_count", 0)),
                        "passage_compact_lock_wp_count=%d" % int(failure.get("passage_compact_lock_wp_count", 0)),
                        "passage_segment_count=%d" % int(failure.get("passage_segment_count", 0)),
                        "passage_hold_wp_count=%d" % int(failure.get("passage_hold_wp_count", 0)),
                        "passage_hold_shape_count=%d" % int(failure.get("passage_hold_shape_count", 0)),
                        "passage_hold_min_arm_margin=%s" % self._fmt(float(failure.get("passage_hold_min_arm_margin", 0.0))),
                        "selected_passage_risk_count=%d" % int(failure.get("selected_passage_risk_count", 0)),
                        "selected_passage_compact_lock_count=%d" % int(failure.get("selected_passage_compact_lock_count", 0)),
                        "release_min=%.3f" % float(failure.get("release_min", 0.0)),
                        "release_max=%.3f" % float(failure.get("release_max", 0.0)),
                        "release_mean=%.3f" % float(failure.get("release_mean", 0.0)),
                        "release_at_failure=%.3f" % float(failure.get("release_at_failure", 0.0)),
                        "transition_repair_inserted=%d" % int(failure.get("transition_repair_inserted", 0)),
                        "swept_repair_inserted=%d" % int(failure.get("swept_repair_inserted", 0)),
                        "local_repair_attempts=%d" % int(failure.get("local_repair_attempts", 0)),
                        "edge_repair_enabled=%s" % str(failure.get("edge_repair_enabled", self.edge_repair_enabled)),
                        "edge_repair_used=%s" % str(failure.get("edge_repair_used", False)),
                        "edge_repair_attempts=%d" % int(failure.get("edge_repair_attempts", 0)),
                        "edge_repair_successes=%d" % int(failure.get("edge_repair_successes", 0)),
                        "edge_repair_inserted_knots=%d" % int(failure.get("edge_repair_inserted_knots", 0)),
                        "edge_repair_min_body_margin=%s" % self._fmt(float(failure.get("edge_repair_min_body_margin", float("inf")))),
                        "edge_repair_min_arm_margin=%s" % self._fmt(float(failure.get("edge_repair_min_arm_margin", float("inf")))),
                        "edge_repair_time_budget=%.2f" % float(failure.get("edge_repair_time_budget", self.edge_repair_time_budget)),
                        "edge_repair_timeout=%s" % str(failure.get("edge_repair_timeout", False)),
                        "terminal_bridge_attempts=%d" % int(failure.get("terminal_bridge_attempts", 0)),
                        "terminal_bridge_successes=%d" % int(failure.get("terminal_bridge_successes", 0)),
                        "terminal_bridge_knots=%d" % int(failure.get("terminal_bridge_knots", 0)),
                        "terminal_bridge_best_status=%s" % failure.get("terminal_bridge_best_status", "none"),
                        "terminal_bridge_best_body_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
                        % (
                            self._fmt(float(failure.get("terminal_bridge_best_body_margin", float("-inf")))),
                            failure.get("terminal_bridge_best_body_obstacle", "none"),
                            failure.get("terminal_bridge_best_body_point", (0.0, 0.0, 0.0))[0],
                            failure.get("terminal_bridge_best_body_point", (0.0, 0.0, 0.0))[1],
                            failure.get("terminal_bridge_best_body_point", (0.0, 0.0, 0.0))[2],
                        ),
                        "terminal_bridge_best_arm_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
                        % (
                            self._fmt(float(failure.get("terminal_bridge_best_arm_margin", float("-inf")))),
                            failure.get("terminal_bridge_best_arm_obstacle", "none"),
                            failure.get("terminal_bridge_best_arm_point", (0.0, 0.0, 0.0))[0],
                            failure.get("terminal_bridge_best_arm_point", (0.0, 0.0, 0.0))[1],
                            failure.get("terminal_bridge_best_arm_point", (0.0, 0.0, 0.0))[2],
                        ),
                        "refine_time_budget=%.2f" % float(failure.get("refine_time_budget", 0.0)),
                        "timeout_sec=%.2f" % float(failure.get("timeout_sec", 0.0)),
                        "reach_search_time_budget=%.2f" % float(failure.get("reach_search_time_budget", 0.0)),
                        "reach_search_timeout_sec=%.2f" % float(failure.get("reach_search_timeout_sec", 0.0)),
                        "reach_search_expanded=%d" % int(failure.get("reach_search_expanded", 0)),
                        "reach_search_max_states=%d" % int(failure.get("reach_search_max_states", 0)),
                    ]
                )
            return "\n".join(lines)
        safety = result["safety"]
        candidate = result["candidate"]
        selection_detail = candidate.get("selection_detail", {})
        reference_repair = result.get("reference_repair", {})
        refinement = result.get("refinement", {})
        interpolation = result.get("interpolation", {})
        lines = [
            "status=%s" % status,
            "planner_version=%s" % PLANNER_VERSION,
            "method=ours_full_min",
            "scenario=%s" % self.scenario_name,
            "freeze_plan_when_ready=%s" % str(self.freeze_plan_when_ready),
            "frozen_plan=%s" % str(self.frozen_plan_result is not None),
            "selected_reach_goal_ready=True",
            "global_path_ready=True",
            "ego_reference_ready=True",
            "ego_reference_points=%d" % int(reference_repair.get("ego_reference_points", 0)),
            "ego_reference_param_cache_used=%s" % str(self.ego_reference_param_cache_used),
            "unsafe_ego_points=%d" % int(reference_repair.get("unsafe_ego_points", 0)),
            "first_unsafe_ego_idx=%d" % int(reference_repair.get("first_unsafe_ego_idx", -1)),
            "first_unsafe_ego_margin=%s obstacle=%s pose=[%.3f, %.3f, %.3f] point=[%.3f, %.3f, %.3f]"
            % (
                self._fmt(float(reference_repair.get("first_unsafe_ego_margin", float("inf")))),
                reference_repair.get("first_unsafe_ego_obstacle", "none"),
                reference_repair.get("first_unsafe_ego_pose", (0.0, 0.0, 0.0))[0],
                reference_repair.get("first_unsafe_ego_pose", (0.0, 0.0, 0.0))[1],
                reference_repair.get("first_unsafe_ego_pose", (0.0, 0.0, 0.0))[2],
                reference_repair.get("first_unsafe_ego_point", (0.0, 0.0, 0.0))[0],
                reference_repair.get("first_unsafe_ego_point", (0.0, 0.0, 0.0))[1],
                reference_repair.get("first_unsafe_ego_point", (0.0, 0.0, 0.0))[2],
            ),
            "global_repair_used=%s" % str(reference_repair.get("global_repair_used", False)),
            "global_repair_points=%d" % int(reference_repair.get("global_repair_points", 0)),
            "global_repair_reason=%s" % reference_repair.get("global_repair_reason", "none"),
            "slot_keyframes_inserted=%d" % int(reference_repair.get("slot_keyframes_inserted", 0)),
            "slot_poses_projected=%d" % int(reference_repair.get("slot_poses_projected", 0)),
            "slot_keyframe_constraints=%s" % ",".join(reference_repair.get("slot_keyframe_constraints", [])),
            "reference_edge_splits=%d" % int(reference_repair.get("reference_edge_splits", 0)),
            "reference_edge_detours=%d" % int(reference_repair.get("reference_edge_detours", 0)),
            "reference_edge_unresolved=%d" % int(reference_repair.get("reference_edge_unresolved", 0)),
            "reference_edge_densify_timeout=%s" % str(reference_repair.get("reference_edge_densify_timeout", False)),
            "reference_edge_min_body_margin=%s" % self._fmt(float(reference_repair.get("reference_edge_min_body_margin", float("inf")))),
            "reference_edge_min_arm_margin=%s" % self._fmt(float(reference_repair.get("reference_edge_min_arm_margin", float("inf")))),
            "reference_keyframes_pruned=%d" % int(reference_repair.get("reference_keyframes_pruned", 0)),
            "reference_keyframes_after_compression=%d" % int(reference_repair.get("reference_keyframes_after_compression", 0)),
            "terminal_tail_pruned=%d" % int(reference_repair.get("terminal_tail_pruned", 0)),
            "independent_skeleton_used=%s" % str(reference_repair.get("independent_skeleton_used", False)),
            "independent_skeleton_points=%d" % int(reference_repair.get("independent_skeleton_points", 0)),
            "body_skeleton_used=%s" % str(reference_repair.get("body_skeleton_used", False)),
            "body_skeleton_status=%s" % reference_repair.get("body_skeleton_status", "none"),
            "body_skeleton_expanded=%d" % int(reference_repair.get("body_skeleton_expanded", 0)),
            "body_skeleton_raw_points=%d" % int(reference_repair.get("body_skeleton_raw_points", 0)),
            "body_skeleton_start_margin=%s" % self._fmt(float(reference_repair.get("body_skeleton_start_margin", float("-inf")))),
            "body_skeleton_goal_margin=%s" % self._fmt(float(reference_repair.get("body_skeleton_goal_margin", float("-inf")))),
            "whole_body_refinement_used=%s" % str(refinement.get("whole_body_refinement_used", False)),
            "refinement_fallback_reason=%s" % refinement.get("fallback_reason", "none"),
            "slot_corridor_fallback_used=%s" % str(refinement.get("slot_corridor_fallback_used", False)),
            "slot_corridor_points=%d" % int(refinement.get("slot_corridor_points", 0)),
            "local_repair_used=%s" % str(refinement.get("local_repair_used", False)),
            "sequence_optimizer=%s" % refinement.get("sequence_optimizer", "none"),
            "candidate_count=%d" % int(refinement.get("candidate_count", 0)),
            "expanded_edges=%d" % int(refinement.get("expanded_edges", 0)),
            "sequence_beam_width=%d" % int(refinement.get("sequence_beam_width", self.sequence_beam_width)),
            "sequence_effective_beam_width=%d" % int(refinement.get("sequence_effective_beam_width", self.sequence_beam_width)),
            "sequence_edge_eval_neighbors=%d" % int(refinement.get("sequence_edge_eval_neighbors", self.sequence_edge_eval_neighbors)),
            "sequence_effective_transition_neighbors=%d" % int(refinement.get("sequence_effective_transition_neighbors", self.sequence_transition_neighbors)),
            "sequence_edge_exhaustive_retries=%d" % int(refinement.get("sequence_edge_exhaustive_retries", 0)),
            "sequence_edge_collision_lazy=%s" % str(refinement.get("sequence_edge_collision_lazy", True)),
            "sequence_edge_safety_in_dp=%s" % str(refinement.get("sequence_edge_safety_in_dp", False)),
            "expanded_candidate_indices=%d" % int(refinement.get("expanded_candidate_indices", 0)),
            "local_repair_window_start=%d" % int(refinement.get("local_repair_window_start", -1)),
            "local_repair_window_end=%d" % int(refinement.get("local_repair_window_end", -1)),
            "sequence_dp_candidates_per_wp=%d" % int(refinement.get("sequence_dp_candidates_per_wp", self.sequence_dp_candidates_per_wp)),
            "sequence_dp_max_candidates_per_wp=%d" % int(refinement.get("sequence_dp_max_candidates_per_wp", self.sequence_dp_candidates_per_wp)),
            "sequence_dp_min_pose_diversity=%d" % int(refinement.get("sequence_dp_min_pose_diversity", 0)),
            "sequence_dp_max_pose_diversity=%d" % int(refinement.get("sequence_dp_max_pose_diversity", 0)),
            "sequence_dp_rich_risk_layers=%d" % int(refinement.get("sequence_dp_rich_risk_layers", 0)),
            "sequence_approach_release_candidate_count=%d" % int(refinement.get("sequence_approach_release_candidate_count", 0)),
            "sequence_approach_bridge_candidate_count=%d" % int(refinement.get("sequence_approach_bridge_candidate_count", 0)),
            "sequence_dp_candidate_count=%d" % int(refinement.get("sequence_dp_candidate_count", 0)),
            "sequence_candidate_budget_sec=%.2f" % float(refinement.get("sequence_candidate_budget_sec", 0.0)),
            "sequence_dp_reserved_sec=%.2f" % float(refinement.get("sequence_dp_reserved_sec", 0.0)),
            "sequence_candidate_generation_sec=%.2f" % float(refinement.get("sequence_candidate_generation_sec", 0.0)),
            "sequence_dp_sec=%.2f" % float(refinement.get("sequence_dp_sec", 0.0)),
            "sequence_dp_partial_timeout=%s" % str(refinement.get("sequence_dp_partial_timeout", False)),
            "sequence_dp_partial_timeout_wp=%d" % int(refinement.get("sequence_dp_partial_timeout_wp", -1)),
            "edge_body_margin_target=%.3f" % float(refinement.get("edge_body_margin_target", self.edge_body_margin_target)),
            "edge_arm_margin_target=%.3f" % float(refinement.get("edge_arm_margin_target", self.edge_arm_margin_target)),
            "sequence_risk_body_margin=%.3f" % float(refinement.get("sequence_risk_body_margin", self.sequence_risk_body_margin)),
            "sequence_risk_arm_margin=%.3f" % float(refinement.get("sequence_risk_arm_margin", self.sequence_risk_arm_margin)),
            "sequence_cost=%.3f" % float(refinement.get("sequence_cost", 0.0)),
            "transition_repair_inserted=%d" % int(refinement.get("transition_repair_inserted", 0)),
            "swept_repair_inserted=%d" % int(refinement.get("swept_repair_inserted", 0)),
            "local_repair_attempts=%d" % int(refinement.get("local_repair_attempts", 0)),
            "edge_repair_enabled=%s" % str(refinement.get("edge_repair_enabled", self.edge_repair_enabled)),
            "edge_repair_used=%s" % str(refinement.get("edge_repair_used", False)),
            "edge_repair_attempts=%d" % int(refinement.get("edge_repair_attempts", 0)),
            "edge_repair_successes=%d" % int(refinement.get("edge_repair_successes", 0)),
            "edge_repair_inserted_knots=%d" % int(refinement.get("edge_repair_inserted_knots", 0)),
            "edge_repair_min_body_margin=%s" % self._fmt(float(refinement.get("edge_repair_min_body_margin", float("inf")))),
            "edge_repair_min_arm_margin=%s" % self._fmt(float(refinement.get("edge_repair_min_arm_margin", float("inf")))),
            "edge_repair_time_budget=%.2f" % float(refinement.get("edge_repair_time_budget", self.edge_repair_time_budget)),
            "edge_repair_timeout=%s" % str(refinement.get("edge_repair_timeout", False)),
            "terminal_bridge_attempts=%d" % int(refinement.get("terminal_bridge_attempts", 0)),
            "terminal_bridge_successes=%d" % int(refinement.get("terminal_bridge_successes", 0)),
            "terminal_bridge_knots=%d" % int(refinement.get("terminal_bridge_knots", 0)),
            "terminal_bridge_best_status=%s" % refinement.get("terminal_bridge_best_status", "none"),
            "terminal_bridge_best_body_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
            % (
                self._fmt(float(refinement.get("terminal_bridge_best_body_margin", float("-inf")))),
                refinement.get("terminal_bridge_best_body_obstacle", "none"),
                refinement.get("terminal_bridge_best_body_point", (0.0, 0.0, 0.0))[0],
                refinement.get("terminal_bridge_best_body_point", (0.0, 0.0, 0.0))[1],
                refinement.get("terminal_bridge_best_body_point", (0.0, 0.0, 0.0))[2],
            ),
            "terminal_bridge_best_arm_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
            % (
                self._fmt(float(refinement.get("terminal_bridge_best_arm_margin", float("-inf")))),
                refinement.get("terminal_bridge_best_arm_obstacle", "none"),
                refinement.get("terminal_bridge_best_arm_point", (0.0, 0.0, 0.0))[0],
                refinement.get("terminal_bridge_best_arm_point", (0.0, 0.0, 0.0))[1],
                refinement.get("terminal_bridge_best_arm_point", (0.0, 0.0, 0.0))[2],
            ),
            "refinement_input_points=%d" % int(refinement.get("refinement_input_points", 0)),
            "pose_refined_count=%d" % int(refinement.get("pose_refined_count", 0)),
            "arm_optimized_count=%d" % int(refinement.get("arm_optimized_count", 0)),
            "max_ego_deviation=%.3f" % float(refinement.get("max_ego_deviation", 0.0)),
            "mean_ego_deviation=%.3f" % float(refinement.get("mean_ego_deviation", 0.0)),
            "max_arm_com_offset=%.3f" % float(refinement.get("max_arm_com_offset", 0.0)),
            "mean_arm_com_offset=%.3f" % float(refinement.get("mean_arm_com_offset", 0.0)),
            "max_system_com_offset=%.3f" % float(safety.get("max_system_com_offset", 0.0)),
            "mean_system_com_offset=%.3f" % float(safety.get("mean_system_com_offset", 0.0)),
            "hard_com_limit=%.3f" % self.hard_com_limit,
            "max_gravity_torque=%.3f" % float(safety.get("max_gravity_torque", 0.0)),
            "mean_gravity_torque=%.3f" % float(safety.get("mean_gravity_torque", 0.0)),
            "tau_limit=%.3f" % self.tau_limit,
            "max_arm_lateral_extent=%.3f" % float(refinement.get("max_arm_lateral_extent", 0.0)),
            "transit_compact_weight=%.3f" % float(refinement.get("transit_compact_weight", self.refine_compact_weight)),
            "transit_com_weight=%.3f" % float(refinement.get("transit_com_weight", self.refine_com_weight)),
            "transit_clearance_weight=%.3f" % float(refinement.get("transit_clearance_weight", self.refine_clearance_weight)),
            "refine_min_arm_margin=%.3f" % float(refinement.get("refine_min_arm_margin", self.refine_min_arm_margin)),
            "required_body_margin=%.3f" % self.required_body_margin,
            "required_arm_margin=%.3f" % self.required_arm_margin,
            "morphology_gate=%s" % refinement.get("morphology_gate", "none"),
            "morphology_home_hold_weight=%.3f" % float(refinement.get("morphology_home_hold_weight", self.morphology_home_hold_weight)),
            "morphology_activation_margin=%.3f" % float(refinement.get("morphology_activation_margin", self.morphology_activation_margin)),
            "morphology_min_active_theta=%.3f" % float(refinement.get("morphology_min_active_theta", self.morphology_min_active_theta)),
            "passage_morphology_gate_enabled=%s" % str(refinement.get("passage_morphology_gate_enabled", self.passage_morphology_gate_enabled)),
            "passage_risk_wp_count=%d" % int(refinement.get("passage_risk_wp_count", 0)),
            "passage_compact_safe_wp_count=%d" % int(refinement.get("passage_compact_safe_wp_count", 0)),
            "passage_compact_lock_wp_count=%d" % int(refinement.get("passage_compact_lock_wp_count", 0)),
            "passage_segment_count=%d" % int(refinement.get("passage_segment_count", 0)),
            "passage_hold_wp_count=%d" % int(refinement.get("passage_hold_wp_count", 0)),
            "passage_hold_shape_count=%d" % int(refinement.get("passage_hold_shape_count", 0)),
            "passage_hold_min_arm_margin=%s" % self._fmt(float(refinement.get("passage_hold_min_arm_margin", 0.0))),
            "selected_passage_risk_count=%d" % int(refinement.get("selected_passage_risk_count", 0)),
            "selected_passage_compact_lock_count=%d" % int(refinement.get("selected_passage_compact_lock_count", 0)),
            "release_min=%.3f" % float(refinement.get("release_min", 0.0)),
            "release_max=%.3f" % float(refinement.get("release_max", 0.0)),
            "release_mean=%.3f" % float(refinement.get("release_mean", 0.0)),
            "morphology_activated_wp_count=%d" % int(refinement.get("morphology_activated_wp_count", 0)),
            "mean_morphology_activation=%.3f" % float(refinement.get("mean_morphology_activation", 0.0)),
            "valid_candidate_target=%d" % int(refinement.get("valid_candidate_target", self.refine_valid_candidate_target)),
            "sequence_min_candidates_per_wp=%d" % int(refinement.get("sequence_min_candidates_per_wp", self.sequence_min_candidates_per_wp)),
            "shape_candidates_per_pose=%d" % int(refinement.get("shape_candidates_per_pose", self.refine_shape_candidates_per_pose)),
            "shape_policy=%s" % refinement.get("shape_policy", "constraint_cost_library"),
            "arm_interpolation_inserted=%d" % int(interpolation.get("arm_interpolation_inserted", 0)),
            "arm_interpolation_skipped_unsafe=%d" % int(interpolation.get("arm_interpolation_skipped_unsafe", 0)),
            "arm_theta_step_limit=%.2f" % float(interpolation.get("theta_limit", self.max_delta_theta)),
            "arm_phi_step_limit=%.2f" % float(interpolation.get("phi_limit", self.max_delta_phi)),
            "reach_candidate_count=%d" % len(self.reach_candidates),
            "selected_candidate=%d" % result["selected_idx"],
            "selected_candidate_body_margin=%s obstacle=%s"
            % (
                self._fmt(float(selection_detail.get("body_margin", float("inf")))),
                selection_detail.get("body_obstacle", "none"),
            ),
            "selected_candidate_goal_arm_margin=%s obstacle=%s"
            % (
                self._fmt(float(selection_detail.get("goal_arm_margin", float("inf")))),
                selection_detail.get("goal_arm_obstacle", "none"),
            ),
            "selected_candidate_reach_swept_arm_margin=%s obstacle=%s"
            % (
                self._fmt(float(selection_detail.get("swept_arm_margin", float("inf")))),
                selection_detail.get("swept_arm_obstacle", "none"),
            ),
            "selected_standoff=[%.3f, %.3f, %.3f]" % selection_detail.get("standoff_xyz", self._pose_xyz(candidate["pose"])),
            "selected_candidate_strict_safe=%s" % str(selection_detail.get("standoff_strict_safe", False)),
            "selected_candidate_hard_safe=%s" % str(selection_detail.get("standoff_hard_safe", False)),
            "strict_reach_candidate_count=%d" % int(selection_detail.get("strict_candidate_count", 0)),
            "hard_safe_reach_candidate_count=%d" % int(selection_detail.get("hard_safe_candidate_count", 0)),
            "selected_candidate_approach_body_margin=%s"
            % self._fmt(float(selection_detail.get("approach_body_margin", float("inf")))),
            "selected_candidate_approach_arm_margin=%s"
            % self._fmt(float(selection_detail.get("approach_arm_margin", float("inf")))),
            "selected_candidate_approach_hard_unsafe=%s" % str(selection_detail.get("approach_hard_unsafe", False)),
            "approach_safe_candidate_count=%d" % int(selection_detail.get("approach_safe_candidate_count", 0)),
            "uav_waypoints=%d" % len(result["poses"]),
            "arm_waypoints=%d" % len(result["arm_shapes"]),
            "standoff_goal=[%.3f, %.3f, %.3f]" % self._pose_xyz(candidate["pose"]),
            "reach_theta_phi=[%.2f, %.2f]" % (candidate["theta_phi"][0], candidate["theta_phi"][1]),
            "tip_error=%.3f" % result["tip_error"],
            "min_body_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
            % (self._fmt(safety["body_margin"]), safety["body_obstacle"], safety["body_point"][0], safety["body_point"][1], safety["body_point"][2]),
            "min_arm_margin=%s obstacle=%s point=[%.3f, %.3f, %.3f]"
            % (self._fmt(safety["arm_margin"]), safety["arm_obstacle"], safety["arm_point"][0], safety["arm_point"][1], safety["arm_point"][2]),
            "dense_swept_validation=%s" % str(safety.get("swept_validation", False)),
            "swept_feasible=%s" % str(safety.get("swept_feasible", True)),
            "min_swept_body_margin=%s obstacle=%s"
            % (self._fmt(float(safety.get("min_swept_body_margin", float("inf")))), safety.get("min_swept_body_obstacle", "none")),
            "min_swept_arm_margin=%s obstacle=%s"
            % (self._fmt(float(safety.get("min_swept_arm_margin", float("inf")))), safety.get("min_swept_arm_obstacle", "none")),
            "swept_sample_count=%d" % int(safety.get("swept_sample_count", 0)),
            "dynamics_feasible=%s" % str(safety.get("dynamics_feasible", True)),
            "arm_motion_feasible=%s" % str(safety.get("arm_motion_feasible", True)),
            "time_scale=%.2f" % float(safety.get("time_scale", 1.0)),
            "max_velocity=%.3f v_max=%.3f" % (float(safety.get("max_velocity", 0.0)), self.v_max),
            "max_acceleration=%.3f a_max=%.3f" % (float(safety.get("max_acceleration", 0.0)), self.a_max),
            "max_jerk=%.3f j_max=%.3f" % (float(safety.get("max_jerk", 0.0)), self.j_max),
            "max_thrust=%.3f thrust_max=%.3f" % (float(safety.get("max_thrust", 0.0)), self.thrust_max),
            "max_tilt_deg=%.2f tilt_max_deg=%.2f" % (math.degrees(float(safety.get("max_tilt", 0.0))), math.degrees(self.tilt_max)),
            "max_theta_dot=%.2f theta_dot_max=%.2f" % (float(safety.get("max_theta_dot", 0.0)), self.theta_dot_max),
            "max_phi_dot=%.2f phi_dot_max=%.2f" % (float(safety.get("max_phi_dot", 0.0)), self.phi_dot_max),
            "max_theta_acc=%.2f theta_acc_max=%.2f" % (float(safety.get("max_theta_acc", 0.0)), self.theta_acc_max),
            "max_phi_acc=%.2f phi_acc_max=%.2f" % (float(safety.get("max_phi_acc", 0.0)), self.phi_acc_max),
            "arm_sample_count=%d" % int(safety.get("arm_sample_count", 0)),
            "arm_below_target_count=%d" % int(safety.get("arm_below_target_count", 0)),
            "arm_below_min_count=%d" % int(safety.get("arm_below_min_count", 0)),
            "max_theta_step=%.2f" % safety["max_theta_step"],
            "max_phi_step=%.2f" % safety["max_phi_step"],
            "theta_total_variation=%.2f" % safety["theta_total_variation"],
            "phi_total_variation=%.2f" % safety["phi_total_variation"],
        ]
        return "\n".join(lines)

    def _safety_text(self, status, result):
        if result is None:
            return "status=%s method=ours_full_min scenario=%s" % (status, self.scenario_name)
        s = result["safety"]
        return (
            "status=%s method=ours_full_min scenario=%s static_cloud_ready=%s voxels=%d "
            "min_body_margin=%s body_obstacle=%s min_arm_margin=%s arm_obstacle=%s "
            "min_swept_body_margin=%s min_swept_arm_margin=%s swept_feasible=%s "
            "dynamics_feasible=%s arm_motion_feasible=%s max_com=%.3f max_tau=%.3f "
            "max_jerk=%.3f max_tilt_deg=%.2f tip_error=%.3f "
            "max_theta_step=%.2f max_phi_step=%.2f theta_total_variation=%.2f phi_total_variation=%.2f"
            % (
                status,
                self.scenario_name,
                str(s["static_cloud_ready"]),
                s["static_cloud_voxels"],
                self._fmt(s["body_margin"]),
                s["body_obstacle"],
                self._fmt(s["arm_margin"]),
                s["arm_obstacle"],
                self._fmt(float(s.get("min_swept_body_margin", float("inf")))),
                self._fmt(float(s.get("min_swept_arm_margin", float("inf")))),
                str(s.get("swept_feasible", True)),
                str(s.get("dynamics_feasible", True)),
                str(s.get("arm_motion_feasible", True)),
                float(s.get("max_system_com_offset", 0.0)),
                float(s.get("max_gravity_torque", 0.0)),
                float(s.get("max_jerk", 0.0)),
                math.degrees(float(s.get("max_tilt", 0.0))),
                result["tip_error"],
                s["max_theta_step"],
                s["max_phi_step"],
                s["theta_total_variation"],
                s["phi_total_variation"],
            )
        )

    def _markers(self, result, status):
        stamp = rospy.Time.now()
        arr = MarkerArray()
        if result is None:
            failure = self.last_failure_info or {}
            if failure:
                fail_marker = Marker()
                marker_header(fail_marker, self.frame_id, stamp, "whole_body_failure_point", 9000, Marker.SPHERE)
                fail_pose = failure.get("failure_pose", (0.0, 0.0, 0.0))
                fail_marker.pose.position = Point(x=fail_pose[0], y=fail_pose[1], z=fail_pose[2])
                fail_marker.pose.orientation.w = 1.0
                fail_marker.scale.x = 0.16
                fail_marker.scale.y = 0.16
                fail_marker.scale.z = 0.16
                fail_marker.color = color(1.0, 0.0, 0.0, 0.95)
                arr.markers.append(fail_marker)
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_arm_collision_envelope", 2, Marker.SPHERE_LIST)
            marker.action = Marker.DELETE
            arr.markers.append(marker)
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_uav_waypoints", 1, Marker.SPHERE_LIST)
            marker.action = Marker.DELETE
            arr.markers.append(marker)
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_arm_tip_path", 3, Marker.LINE_STRIP)
            marker.action = Marker.DELETE
            arr.markers.append(marker)
            for idx in range(self.last_marker_arm_count):
                marker = Marker()
                marker_header(marker, self.frame_id, stamp, "whole_body_arm_waypoints", idx + 100, Marker.LINE_STRIP)
                marker.action = Marker.DELETE
                arr.markers.append(marker)
            self.last_marker_arm_count = 0
            for idx in range(self.last_interpolated_arm_marker_count):
                marker = Marker()
                marker_header(marker, self.frame_id, stamp, "whole_body_arm_interpolated", idx + 2000, Marker.LINE_STRIP)
                marker.action = Marker.DELETE
                arr.markers.append(marker)
            self.last_interpolated_arm_marker_count = 0
            for idx in range(self.last_low_clearance_marker_count):
                marker = Marker()
                marker_header(marker, self.frame_id, stamp, "whole_body_arm_low_clearance", idx + 500, Marker.SPHERE)
                marker.action = Marker.DELETE
                arr.markers.append(marker)
            self.last_low_clearance_marker_count = 0
            for idx in range(self.last_waypoint_label_count):
                marker = Marker()
                marker_header(marker, self.frame_id, stamp, "whole_body_waypoint_index", idx + 1000, Marker.TEXT_VIEW_FACING)
                marker.action = Marker.DELETE
                arr.markers.append(marker)
            self.last_waypoint_label_count = 0
            return arr

        def timeline_color(idx, count, alpha=0.85):
            denom = float(max(count - 1, 1))
            t = min(1.0, max(0.0, float(idx) / denom))
            if t < 0.5:
                u = t * 2.0
                return color(0.10 + 0.90 * u, 0.95, 1.00 - 0.85 * u, alpha)
            u = (t - 0.5) * 2.0
            return color(1.00, 0.95 - 0.80 * u, 0.15 + 0.65 * u, alpha)

        line = Marker()
        marker_header(line, self.frame_id, stamp, "whole_body_uav_path", 0, Marker.LINE_STRIP)
        line.pose.orientation.w = 1.0
        line.scale.x = 0.05
        line.color = color(0.0, 1.0, 0.2, 0.95) if status == "WHOLE_BODY_PLAN_READY" else color(1.0, 0.2, 0.0, 0.95)
        line.points = [point_msg(p.position.x, p.position.y, p.position.z) for p in result["poses"]]
        arr.markers.append(line)

        waypoint_dots = Marker()
        marker_header(waypoint_dots, self.frame_id, stamp, "whole_body_uav_waypoints", 1, Marker.SPHERE_LIST)
        waypoint_dots.pose.orientation.w = 1.0
        waypoint_dots.scale.x = 0.055
        waypoint_dots.scale.y = 0.055
        waypoint_dots.scale.z = 0.055
        for idx, pose in enumerate(result["poses"]):
            waypoint_dots.points.append(point_msg(pose.position.x, pose.position.y, pose.position.z))
            waypoint_dots.colors.append(timeline_color(idx, len(result["poses"]), 0.95))
        arr.markers.append(waypoint_dots)

        clear_failure = Marker()
        marker_header(clear_failure, self.frame_id, stamp, "whole_body_failure_point", 9000, Marker.SPHERE)
        clear_failure.action = Marker.DELETE
        arr.markers.append(clear_failure)

        target_marker = Marker()
        marker_header(target_marker, self.frame_id, stamp, "whole_body_target", 10, Marker.SPHERE)
        target_marker.pose.position = point_msg(*self.target["pose"][:3])
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.10
        target_marker.scale.y = 0.10
        target_marker.scale.z = 0.10
        target_marker.color = color(1.0, 0.0, 1.0, 1.0)
        arr.markers.append(target_marker)

        standoff_marker = Marker()
        marker_header(standoff_marker, self.frame_id, stamp, "whole_body_selected_standoff", 11, Marker.ARROW)
        standoff_marker.pose = result["candidate"]["pose"]
        standoff_marker.scale.x = 0.35
        standoff_marker.scale.y = 0.06
        standoff_marker.scale.z = 0.06
        standoff_marker.color = color(0.0, 1.0, 0.1, 1.0)
        arr.markers.append(standoff_marker)

        marker_poses, marker_shapes, marker_progress = self._visual_arm_samples(
            result["poses"], result["arm_shapes"]
        )

        arm_marker_count = 0
        tip_path = Marker()
        marker_header(tip_path, self.frame_id, stamp, "whole_body_arm_tip_path", 3, Marker.LINE_STRIP)
        tip_path.pose.orientation.w = 1.0
        tip_path.scale.x = 0.035
        for progress, pose, shape in zip(marker_progress, marker_poses, marker_shapes):
            pts = world_arm_points(pose, shape[0], shape[1], self.arm_params)
            if not pts:
                continue
            tip_path.points.append(Point(x=pts[-1][0], y=pts[-1][1], z=pts[-1][2]))
            tip_path.colors.append(timeline_color(progress, len(result["poses"]), 0.95))
        arr.markers.append(tip_path)

        for progress, pose, shape in zip(marker_progress, marker_poses, marker_shapes):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_arm_interpolated", arm_marker_count + 2000, Marker.LINE_STRIP)
            marker.scale.x = max(self.arm_params["radius"], 0.012)
            marker.color = timeline_color(progress, len(result["poses"]), 0.28)
            marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in world_arm_points(pose, shape[0], shape[1], self.arm_params)]
            arr.markers.append(marker)
            arm_marker_count += 1
        for idx in range(arm_marker_count, self.last_interpolated_arm_marker_count):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_arm_interpolated", idx + 2000, Marker.LINE_STRIP)
            marker.action = Marker.DELETE
            arr.markers.append(marker)
        self.last_interpolated_arm_marker_count = arm_marker_count

        arm_marker_count = 0
        for marker_idx, (pose, shape) in enumerate(zip(result["poses"], result["arm_shapes"])):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_arm_waypoints", arm_marker_count + 100, Marker.LINE_STRIP)
            marker.scale.x = max(self.arm_params["radius"] * 2.0, 0.02)
            marker.color = timeline_color(marker_idx, len(result["poses"]), 0.92)
            marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in world_arm_points(pose, shape[0], shape[1], self.arm_params)]
            arr.markers.append(marker)
            arm_marker_count += 1
        for idx in range(arm_marker_count, self.last_marker_arm_count):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_arm_waypoints", idx + 100, Marker.LINE_STRIP)
            marker.action = Marker.DELETE
            arr.markers.append(marker)
        self.last_marker_arm_count = arm_marker_count

        label_count = 0
        for marker_idx, (pose, shape) in enumerate(zip(result["poses"], result["arm_shapes"])):
            if (
                marker_idx != 0
                and marker_idx != len(result["poses"]) - 1
                and marker_idx % self.marker_waypoint_label_stride != 0
            ):
                continue
            arm_points = world_arm_points(pose, shape[0], shape[1], self.arm_params)
            world_direction = "--"
            if len(arm_points) >= 2:
                dx = arm_points[-1][0] - arm_points[0][0]
                dy = arm_points[-1][1] - arm_points[0][1]
                if math.hypot(dx, dy) > 0.02:
                    world_direction = "%.0f" % math.degrees(math.atan2(dy, dx))
            label = Marker()
            marker_header(label, self.frame_id, stamp, "whole_body_waypoint_index", label_count + 1000, Marker.TEXT_VIEW_FACING)
            label.pose.position = point_msg(pose.position.x, pose.position.y, pose.position.z + 0.16)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.075
            label.color = timeline_color(marker_idx, len(result["poses"]), 1.0)
            label.text = "%03d th=%.0f ph=%.0f\nyaw=%.0f dir=%s" % (
                marker_idx,
                shape[0],
                shape[1],
                math.degrees(self._pose_yaw(pose)),
                world_direction,
            )
            arr.markers.append(label)
            label_count += 1
        for idx in range(label_count, self.last_waypoint_label_count):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_waypoint_index", idx + 1000, Marker.TEXT_VIEW_FACING)
            marker.action = Marker.DELETE
            arr.markers.append(marker)
        self.last_waypoint_label_count = label_count

        low_marker_count = 0
        arm_required = self.arm_params["radius"] + self.arm_clearance
        envelope = Marker()
        marker_header(envelope, self.frame_id, stamp, "whole_body_arm_collision_envelope", 2, Marker.SPHERE_LIST)
        envelope.pose.orientation.w = 1.0
        envelope.scale.x = 2.0 * arm_required
        envelope.scale.y = 2.0 * arm_required
        envelope.scale.z = 2.0 * arm_required
        envelope.color = color(1.0, 0.35, 0.0, 0.22)
        for pose, shape in zip(result["poses"], result["arm_shapes"]):
            for p in world_arm_points(pose, shape[0], shape[1], self.arm_params):
                envelope.points.append(Point(x=p[0], y=p[1], z=p[2]))
                margin, _, _ = self._point_margin(p, arm_required, False)
                if margin >= self.refine_clearance_target:
                    continue
                marker = Marker()
                marker_header(marker, self.frame_id, stamp, "whole_body_arm_low_clearance", low_marker_count + 500, Marker.SPHERE)
                marker.pose.position = Point(x=p[0], y=p[1], z=p[2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.07
                marker.scale.y = 0.07
                marker.scale.z = 0.07
                marker.color = color(1.0, 0.05, 0.0, 0.95) if margin < self.refine_min_arm_margin else color(1.0, 0.55, 0.0, 0.75)
                arr.markers.append(marker)
                low_marker_count += 1
        arr.markers.append(envelope)
        for idx in range(low_marker_count, self.last_low_clearance_marker_count):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "whole_body_arm_low_clearance", idx + 500, Marker.SPHERE)
            marker.action = Marker.DELETE
            arr.markers.append(marker)
        self.last_low_clearance_marker_count = low_marker_count
        return arr

    def _publish_plan_result(self, result, status):
        self.plan_pub.publish(String(data=self._plan_text(status, result)))
        self.safety_pub.publish(String(data=self._safety_text(status, result)))
        self.marker_pub.publish(self._markers(result, status))
        if result is not None and status == "WHOLE_BODY_PLAN_READY":
            self.path_pub.publish(self._path_msg(result["poses"]))
            self.arm_pub.publish(self._arm_msg(result["arm_shapes"]))

    def _plan_once(self):
        if self.freeze_plan_when_ready and self.frozen_plan_result is not None:
            self._publish_plan_result(self.frozen_plan_result, self.frozen_plan_status)
            return self.frozen_plan_status
        self.plan_pub.publish(
            String(
                data=(
                    "status=PLANNING_WHOLE_BODY_REFRESH_INPUTS\n"
                    "planner_version=%s\n"
                    "method=ours_full_min\n"
                    "scenario=%s\n"
                    "input_refresh_attempts=%d\n"
                    "global_path_received=%s\n"
                    "ego_reference_received=%s\n"
                    "reach_candidate_count=%d\n"
                    "static_cloud_esdf=%s"
                )
                % (
                    PLANNER_VERSION,
                    self.scenario_name,
                    self.input_refresh_attempts,
                    str(self.global_path is not None and bool(self.global_path.poses)),
                    str(self.reference_path is not None and bool(self.reference_path.poses)),
                    len(self.reach_candidates),
                    getattr(self.static_occupancy, "distance_field_status", "unknown"),
                )
            )
        )
        try:
            result, status = self._build_plan()
        except Exception as exc:
            rospy.logerr("whole_body_planner_node: build_plan exception: %s", exc)
            self.plan_pub.publish(
                String(
                    data=(
                        "status=WHOLE_BODY_EXCEPTION\n"
                        "planner_version=%s\n"
                        "method=ours_full_min\n"
                        "scenario=%s\n"
                        "error=%s"
                    )
                    % (PLANNER_VERSION, self.scenario_name, str(exc))
                )
            )
            return "WHOLE_BODY_EXCEPTION"
        if self.freeze_plan_when_ready and result is not None and status == "WHOLE_BODY_PLAN_READY":
            self.frozen_plan_result = result
            self.frozen_plan_status = status
        self._publish_plan_result(result, status)
        return status

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate_hz", 2.0)))
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - self.last_plan_time).to_sec() >= self.plan_period:
                self.last_plan_time = now
                status = self._plan_once()
                if status not in ("WHOLE_BODY_PLAN_READY", "WAITING_CURRENT_POSE", "WAITING_GLOBAL_PATH", "WAITING_EGO_REFERENCE", "WAITING_REACH_CANDIDATES"):
                    self.last_plan_time = rospy.Time.now() + rospy.Duration(max(0.0, self.failure_hold_time))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("whole_body_planner_node")
    try:
        WholeBodyPlannerNode().spin()
    except rospy.ROSInterruptException:
        pass
