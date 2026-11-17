#!/usr/bin/env python3
"""Record or replay the planner-visible SLAM map without Gazebo truth access."""

import json
from collections import deque
from pathlib import Path
import sys
import threading
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger, TriggerResponse

from uav_continuum_whole_body_v2.mapping import IncrementalVoxelMapper, MapSnapshot


class SlamMapSnapshotNode:
    def __init__(self):
        self.mode = str(rospy.get_param("~mode", "replay")).strip().lower()
        if self.mode not in ("record", "replay"):
            raise ValueError("~mode must be record or replay")
        self.path = Path(rospy.get_param("~snapshot_file", str(
            PACKAGE / "generated" / "scenario_3c_frozen_slam_map.npz"
        )))
        self.frame_id = str(rospy.get_param("~frame_id", "world"))
        self.localization_source = str(rospy.get_param(
            "~localization_source", "unspecified"
        ))
        self.formal_experiment_allowed = bool(rospy.get_param(
            "~formal_experiment_allowed", False
        ))
        if self.mode == "record" and self.formal_experiment_allowed:
            raise ValueError(
                "recorded maps cannot self-declare formal use; run audit_slam_map.py"
            )
        self.map_source = str(rospy.get_param("~map_source", self.mode + "_slam"))
        self.publish_rate = max(0.001, float(rospy.get_param("~publish_rate_hz", 1.0)))
        self.publish_during_record = bool(rospy.get_param(
            "~publish_during_record", True
        ))
        self.maximum_cloud_points = max(100, int(rospy.get_param(
            "~maximum_cloud_points", 20000
        )))
        self.minimum_cloud_interval = max(0.0, float(rospy.get_param(
            "~minimum_cloud_interval_s", 0.0
        )))
        self.minimum_free_ray_interval = max(0.0, float(rospy.get_param(
            "~minimum_free_ray_interval_s", 2.0
        )))
        self.footprint_radius_xy = max(0.0, float(rospy.get_param(
            "~footprint_clear_radius_xy", 0.0
        )))
        self.footprint_half_height = max(0.0, float(rospy.get_param(
            "~footprint_clear_half_height", 0.0
        )))
        self.last_integrated_stamps = {}
        self.last_free_integrated_stamps = {}
        self.lock = threading.Lock()
        self.clearing_lock = threading.Lock()
        # Timer, explicit Trigger and shutdown may all request a compressed
        # snapshot. Only one may build/write it at a time; concurrent saves of
        # million-voxel maps can otherwise starve or break the Trigger call.
        self.save_lock = threading.Lock()
        self.origin = None
        self.origin_frame = None
        self.origin_history = deque(maxlen=200)
        self.clearing_path = deque(maxlen=10000)
        self.survey_centerline = []
        self.snapshot = None
        self.dirty = False
        self.last_saved_hash = None
        self.occupied_pub = rospy.Publisher(
            "/planner/map/occupied_voxels", PointCloud2, queue_size=1, latch=True
        )
        self.observed_pub = rospy.Publisher(
            "/planner/map/observed_voxels", PointCloud2, queue_size=1, latch=True
        )
        self.metadata_pub = rospy.Publisher(
            "/planner/map/metadata", String, queue_size=1, latch=True
        )
        if self.mode == "replay":
            if not self.path.is_file():
                raise FileNotFoundError("SLAM map snapshot does not exist: %s" % self.path)
            self.snapshot = MapSnapshot.load(self.path)
            self._audit_localization_claim(self.snapshot)
        else:
            self.mapper = IncrementalVoxelMapper(
                resolution=float(rospy.get_param("~resolution", 0.05)),
                maximum_range=float(rospy.get_param("~maximum_range", 8.0)),
                ray_step_ratio=float(rospy.get_param("~ray_step_ratio", 0.5)),
            )
            cloud_topics = rospy.get_param(
                "~registered_cloud_topics",
                [rospy.get_param("~registered_cloud_topic", "/slam/registered_cloud")],
            )
            free_topics = rospy.get_param(
                "~registered_free_ray_topics",
                [rospy.get_param(
                    "~registered_free_ray_topic", "/slam/registered_free_ray_endpoints"
                )],
            )
            if isinstance(cloud_topics, str):
                cloud_topics = [cloud_topics]
            if isinstance(free_topics, str):
                free_topics = [free_topics]
            self.registered_cloud_topics = [str(topic) for topic in cloud_topics]
            self.registered_free_ray_topics = [str(topic) for topic in free_topics]
            for topic in self.registered_cloud_topics:
                rospy.Subscriber(topic, PointCloud2, self._cloud_cb, queue_size=1)
            for topic in self.registered_free_ray_topics:
                rospy.Subscriber(topic, PointCloud2, self._free_cloud_cb, queue_size=1)
            rospy.Subscriber(
                rospy.get_param("~sensor_pose_topic", "/slam/sensor_pose"),
                PoseStamped, self._pose_cb, queue_size=10,
            )
            if self.footprint_radius_xy > 0.0:
                rospy.Subscriber(
                    rospy.get_param(
                        "~clearing_pose_topic", "/slam/estimator_body_pose"
                    ),
                    PoseStamped, self._clearing_pose_cb, queue_size=1000,
                )
        self.save_service = rospy.Service(
            "~save_snapshot", Trigger, self._save_snapshot_cb
        )
        rospy.on_shutdown(self._save_if_needed)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._timer_cb)
        rospy.loginfo(
            "slam_map_snapshot_node: mode=%s file=%s localization_source=%s",
            self.mode, self.path, self.localization_source,
        )

    def _audit_localization_claim(self, snapshot):
        stored = str(snapshot.metadata.get("localization_source", "unspecified"))
        if self.localization_source != "unspecified" and stored != self.localization_source:
            raise ValueError(
                "snapshot localization_source=%s does not match requested %s" % (
                    stored, self.localization_source,
                )
            )

    def _pose_cb(self, message):
        frame = message.header.frame_id or self.frame_id
        if frame != self.frame_id:
            rospy.logerr_throttle(
                2.0, "sensor pose must already be registered in %s, got %s",
                self.frame_id, frame,
            )
            return
        position = message.pose.position
        with self.lock:
            self.origin = np.asarray([position.x, position.y, position.z], dtype=float)
            self.origin_frame = frame
            self.origin_history.append((message.header.stamp.to_sec(), self.origin.copy()))

    def _origin_for_stamp(self, stamp):
        with self.lock:
            history = list(self.origin_history)
        if not history:
            return None
        _, origin = min(history, key=lambda item: abs(item[0] - stamp))
        return origin.copy()

    def _cloud_cb(self, message):
        if rospy.is_shutdown():
            return
        cloud_stamp = message.header.stamp.to_sec()
        connection = message._connection_header or {}
        source_key = (
            str(connection.get("topic", "unknown")),
            str(connection.get("callerid", "unknown")),
        )
        with self.lock:
            previous_stamp = self.last_integrated_stamps.get(source_key)
            if (previous_stamp is not None
                    and cloud_stamp - previous_stamp
                    < self.minimum_cloud_interval - 1e-9):
                return
            self.last_integrated_stamps[source_key] = cloud_stamp
        frame = message.header.frame_id or self.frame_id
        if frame != self.frame_id:
            rospy.logerr_throttle(
                2.0, "point cloud must be SLAM-registered in %s, got %s",
                self.frame_id, frame,
            )
            return
        origin = self._origin_for_stamp(cloud_stamp)
        if origin is None:
            rospy.logwarn_throttle(2.0, "waiting for registered sensor pose")
            return
        points = np.asarray([
            row[:3] for row in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ], dtype=float).reshape((-1, 3))
        if not len(points):
            return
        stride = max(1, int(np.ceil(float(len(points)) / self.maximum_cloud_points)))
        clearing_path = []
        traversed_positions = []
        if self.footprint_radius_xy > 0.0:
            with self.clearing_lock:
                clearing_path = list(self.clearing_path)
                traversed_positions = list(self.survey_centerline)
                if clearing_path:
                    self.clearing_path.clear()
                    self.clearing_path.append(clearing_path[-1])
        with self.lock:
            occupied_before = len(self.mapper.occupied)
            accepted = self.mapper.integrate(points, origin, stride=stride)
            occupied_after_hits = len(self.mapper.occupied)
            high_after_hits = sum(
                (index[2] + 0.5) * self.mapper.resolution >= 0.20
                for index in self.mapper.occupied
            )
            if accepted and clearing_path:
                maximum_step = 0.5 * self.mapper.resolution
                swept_positions = [clearing_path[0]]
                for first, second in zip(clearing_path[:-1], clearing_path[1:]):
                    distance = float(np.linalg.norm(second - first))
                    count = max(1, int(np.ceil(distance / maximum_step)))
                    swept_positions.extend(
                        first + ratio * (second - first)
                        for ratio in np.linspace(0.0, 1.0, count + 1)[1:]
                    )
                for center in swept_positions:
                    self.mapper.mark_robot_footprint_free(
                        center, self.footprint_radius_xy,
                        self.footprint_half_height,
                    )
            if accepted and self.publish_during_record:
                self.snapshot = self.mapper.snapshot(
                    frame_id=self.frame_id, source=self.map_source,
                    stamp=message.header.stamp.to_sec(),
                    metadata=self._record_metadata(),
                    traversed_positions=traversed_positions,
                )
            if accepted:
                self.dirty = True
            occupied_after_clear = len(self.mapper.occupied)
            high_after_clear = sum(
                (index[2] + 0.5) * self.mapper.resolution >= 0.20
                for index in self.mapper.occupied
            )
        if high_after_clear < high_after_hits:
            path_array = np.asarray(clearing_path, dtype=float).reshape((-1, 3))
            rospy.logwarn(
                "footprint clearing removed %d elevated occupied voxels; "
                "path_xyz_min=%s path_xyz_max=%s",
                high_after_hits - high_after_clear,
                np.min(path_array, axis=0).tolist() if len(path_array) else [],
                np.max(path_array, axis=0).tolist() if len(path_array) else [],
            )
        rospy.loginfo_throttle(
            5.0,
            "map integration source=%s input_high=%d accepted=%d "
            "occupied=%d->%d->%d elevated=%d->%d observed=%d",
            source_key[1], int(np.count_nonzero(points[:, 2] >= 0.20)), accepted,
            occupied_before, occupied_after_hits, occupied_after_clear,
            high_after_hits, high_after_clear, len(self.mapper.observed),
        )

    def _free_cloud_cb(self, message):
        if rospy.is_shutdown():
            return
        cloud_stamp = message.header.stamp.to_sec()
        connection = message._connection_header or {}
        source_key = (
            str(connection.get("topic", "unknown")),
            str(connection.get("callerid", "unknown")),
        )
        with self.lock:
            previous_stamp = self.last_free_integrated_stamps.get(source_key)
            if (previous_stamp is not None
                    and cloud_stamp - previous_stamp
                    < self.minimum_free_ray_interval - 1e-9):
                return
            self.last_free_integrated_stamps[source_key] = cloud_stamp
        origin = self._origin_for_stamp(cloud_stamp)
        if origin is None:
            return
        frame = message.header.frame_id or self.frame_id
        if frame != self.frame_id:
            rospy.logerr_throttle(
                2.0, "free ray endpoints must be in %s, got %s", self.frame_id, frame
            )
            return
        endpoints = np.asarray([
            row[:3] for row in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ], dtype=float).reshape((-1, 3))
        if not len(endpoints):
            return
        with self.lock:
            self.mapper.integrate_free_rays(endpoints, origin)
            # The corresponding occupied-return frame rebuilds the immutable
            # snapshot. Until then the previous complete snapshot stays valid.
            self.dirty = True

    def _clearing_pose_cb(self, message):
        frame = message.header.frame_id or self.frame_id
        if frame != self.frame_id:
            rospy.logerr_throttle(
                2.0, "footprint pose must be in %s, got %s", self.frame_id, frame
            )
            return
        position = message.pose.position
        center = np.asarray([position.x, position.y, position.z], dtype=float)
        changed = False
        with self.clearing_lock:
            # This callback never touches the heavy voxel sets. Preserve the
            # measured estimator polyline even while a depth frame is being
            # integrated; _cloud_cb later clears its swept physical volume.
            if (not self.clearing_path
                    or float(np.linalg.norm(center - self.clearing_path[-1]))
                    >= 0.25 * self.mapper.resolution):
                self.clearing_path.append(center)
                changed = True
            if (not self.survey_centerline
                    or float(np.linalg.norm(center - self.survey_centerline[-1]))
                    >= 0.5 * self.mapper.resolution):
                self.survey_centerline.append(center)
                changed = True
        if changed:
            with self.lock:
                self.dirty = True

    @staticmethod
    def _cloud_message(points, frame_id, stamp):
        header = Header(stamp=stamp, frame_id=frame_id)
        return point_cloud2.create_cloud_xyz32(header, np.asarray(points, dtype=float).tolist())

    def _metadata(self, snapshot):
        return {
            "state": "READY",
            "map_hash": snapshot.map_hash,
            "source": snapshot.source,
            "localization_source": snapshot.metadata.get("localization_source", "unspecified"),
            "frame_id": snapshot.frame_id,
            "stamp": snapshot.stamp,
            "resolution": snapshot.resolution,
            "occupied_voxels": len(snapshot.occupied_indices),
            "observed_voxels": len(snapshot.observed_indices),
            "unknown_is_occupied": True,
            "truth_geometry_access": False,
            "snapshot_file": str(self.path),
            "snapshot_metadata": snapshot.metadata,
        }

    def _record_metadata(self):
        return {
            "localization_source": self.localization_source,
            "registered_cloud_topics": self.registered_cloud_topics,
            "minimum_cloud_interval_s": self.minimum_cloud_interval,
            "maximum_cloud_points": self.maximum_cloud_points,
            "publish_during_record": self.publish_during_record,
            "footprint_clear_radius_xy": self.footprint_radius_xy,
            "footprint_clear_half_height": self.footprint_half_height,
            "footprint_clear_sync": "measured_pose_sweep",
            "footprint_sweep_step_m": 0.5 * self.mapper.resolution,
            "traversed_centerline_source": "onboard_estimator",
            "unknown_is_occupied": True,
            "truth_derived": False,
            "formal_experiment_allowed": False,
        }

    def _take_pending_footprint(self):
        with self.clearing_lock:
            path = list(self.clearing_path)
            traversed = list(self.survey_centerline)
            if path:
                self.clearing_path.clear()
                self.clearing_path.append(path[-1])
        return path, traversed

    def _apply_footprint_locked(self, path):
        if not path:
            return
        maximum_step = 0.5 * self.mapper.resolution
        swept = [path[0]]
        for first, second in zip(path[:-1], path[1:]):
            distance = float(np.linalg.norm(second - first))
            count = max(1, int(np.ceil(distance / maximum_step)))
            swept.extend(
                first + ratio * (second - first)
                for ratio in np.linspace(0.0, 1.0, count + 1)[1:]
            )
        for center in swept:
            self.mapper.mark_robot_footprint_free(
                center, self.footprint_radius_xy, self.footprint_half_height
            )

    def _save_if_needed(self, force=False):
        with self.save_lock:
            return self._save_if_needed_serialized(force=force)

    def _save_if_needed_serialized(self, force=False):
        if self.mode == "record":
            # The high-rate mapping callback mutates compact voxel sets only.
            # Build the immutable, sorted snapshot here at low rate and always
            # consume the latest estimator polyline first.  This prevents a
            # large map copy/publication from starving the survey pose stream
            # and ensures certification sees the complete measured route.
            with self.lock:
                dirty = self.dirty or bool(force)
                if dirty:
                    # Clear before snapshot construction. Any mapping/pose
                    # callback arriving while the file is being written sets
                    # it again and is therefore preserved for the next save.
                    self.dirty = False
            if not dirty:
                return
            path, traversed = self._take_pending_footprint()
            with self.lock:
                self._apply_footprint_locked(path)
                snapshot = self.mapper.snapshot(
                    frame_id=self.frame_id, source=self.map_source,
                    stamp=rospy.Time.now().to_sec(),
                    metadata=self._record_metadata(),
                    traversed_positions=traversed,
                )
                self.snapshot = snapshot
        else:
            with self.lock:
                snapshot = self.snapshot
                dirty = self.dirty
        if snapshot is None or not dirty or snapshot.map_hash == self.last_saved_hash:
            return
        try:
            snapshot.save(self.path)
        except Exception:
            with self.lock:
                self.dirty = True
            raise
        self.last_saved_hash = snapshot.map_hash
        rospy.loginfo("saved SLAM map snapshot %s hash=%s", self.path, snapshot.map_hash)

    def _save_snapshot_cb(self, _request):
        if self.mode != "record":
            return TriggerResponse(success=False, message="node is not in record mode")
        try:
            self._save_if_needed(force=True)
            if not self.path.is_file():
                raise RuntimeError("snapshot file was not created")
            snapshot = MapSnapshot.load(self.path)
            return TriggerResponse(
                success=True,
                message="saved map hash=%s traversed_positions=%d" % (
                    snapshot.map_hash, len(snapshot.traversed_positions)
                ),
            )
        except Exception as error:
            rospy.logerr("explicit map snapshot failed: %s", error)
            return TriggerResponse(success=False, message=str(error))

    def _timer_cb(self, _event):
        if rospy.is_shutdown():
            return
        if self.mode == "record" and not self.publish_during_record:
            self._save_if_needed()
            return
        with self.lock:
            snapshot = self.snapshot
        if snapshot is None:
            self.metadata_pub.publish(String(data=json.dumps({
                "state": "WAITING_REGISTERED_MAP",
                "truth_geometry_access": False,
                "localization_source": self.localization_source,
            }, sort_keys=True)))
            return
        stamp = rospy.Time.now()
        self.occupied_pub.publish(self._cloud_message(
            snapshot.occupied_points, snapshot.frame_id, stamp
        ))
        self.observed_pub.publish(self._cloud_message(
            snapshot.observed_points, snapshot.frame_id, stamp
        ))
        self.metadata_pub.publish(String(data=json.dumps(
            self._metadata(snapshot), sort_keys=True
        )))
        if self.mode == "record":
            self._save_if_needed()


if __name__ == "__main__":
    rospy.init_node("slam_map_snapshot_node")
    SlamMapSnapshotNode()
    rospy.spin()
