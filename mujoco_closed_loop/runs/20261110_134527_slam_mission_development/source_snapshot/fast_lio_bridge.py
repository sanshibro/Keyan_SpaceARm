"""MuJoCo first-return LiDAR and IMU publisher for the ROS1 FAST-LIO pilot.

The pilot moves a kinematic sensor rig through the existing mission room to
check the ROS message contract. Ground truth is written only after the run for
evaluation; the published messages contain no world pose.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from xml.etree.ElementTree import fromstring, tostring

import mujoco
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs import point_cloud2

from model import model_xml
from mission_scene import MissionScene
from sensors import SurfaceLidar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src'))
from uav_continuum_whole_body_v2.mapping import IncrementalVoxelMapper


FIELDS = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('intensity', 12, PointField.FLOAT32, 1),
    PointField('time', 16, PointField.FLOAT32, 1),
    PointField('ring', 20, PointField.UINT16, 1),
]
POINT_DTYPE = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'time', 'ring'],
                        'formats': ['<f4', '<f4', '<f4', '<f4', '<f4', '<u2'],
                        'offsets': [0, 4, 8, 12, 16, 20], 'itemsize': 24})


def cloud_message(points, stamp, scan_duration=.1, scan_lines=16, vertical_fov=60.):
    """All rays are acquired at the same MuJoCo state, at the scan end time."""
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = 'lidar'
    msg.height = 1
    msg.width = len(points)
    msg.fields = FIELDS
    msg.is_bigendian = False
    msg.point_step = POINT_DTYPE.itemsize
    msg.row_step = len(points) * msg.point_step
    msg.is_dense = True
    packed = np.zeros(len(points), dtype=POINT_DTYPE)
    for col, name in enumerate(('x', 'y', 'z')):
        packed[name] = points[:, col]
    packed['intensity'] = 1.
    # FAST-LIO's Velodyne decoder expects seconds and converts to ms.
    packed['time'] = scan_duration
    elevation = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))
    half=np.deg2rad(vertical_fov/2)
    packed['ring'] = np.clip(np.rint((elevation + half) / (2*half) * (scan_lines-1)), 0, scan_lines-1).astype(np.uint16)
    msg.data = packed.tobytes()
    return msg


def imu_message(stamp, specific_force, angular_velocity):
    msg = Imu()
    msg.header.stamp = stamp
    msg.header.frame_id = 'uav_imu'
    msg.orientation_covariance[0] = -1.  # No attitude supplied to SLAM.
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, angular_velocity)
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, specific_force)
    return msg


def pilot_state(t):
    """Smooth sensor trajectory; stationary first so FAST-LIO can initialize."""
    tau = max(0., t-3.)
    # One translation then lateral translation; derivatives are analytic.
    x = -.6 + .22 * (1.-np.cos(.7*tau))
    y = -1.2 + .18 * (1.-np.cos(.5*tau))
    vx = .154*np.sin(.7*tau) if tau else 0.
    vy = .09*np.sin(.5*tau) if tau else 0.
    ax = .1078*np.cos(.7*tau) if tau else 0.
    ay = .045*np.cos(.5*tau) if tau else 0.
    yaw = .15*np.sin(.25*tau)
    yaw_rate = .0375*np.cos(.25*tau) if tau else 0.
    return np.array([x, y, 1.5]), np.array([vx, vy, 0.]), np.array([ax, ay, 0.]), yaw, yaw_rate


def run_pilot(seconds, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = fromstring(model_xml(False, camera_profile='grasp_visible', nav_camera_profile='front_plus_survey'))
    MissionScene().add(root)
    model = mujoco.MjModel.from_xml_string(tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    uid = model.body('uav').id
    lidar = SurfaceLidar(seed=11, azimuth_count=360, elevation_count=16, vertical_fov=120.)
    rospy.init_node('mujoco_fast_lio_pilot', anonymous=True)
    cloud_pub = rospy.Publisher('/velodyne_points', PointCloud2, queue_size=4)
    imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=100)
    odometry = []
    registered = []
    poses = []
    mapper = IncrementalVoxelMapper(.05, 6.)
    map_stats = {'last_stamp': -1., 'integrated_scans': 0, 'integrated_hits': 0}
    def on_odometry(message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.array([q.w, q.x, q.y, q.z]))
        twist=message.twist.twist.linear
        velocity_world=rotation.reshape(3,3)@np.array([twist.x,twist.y,twist.z])
        odometry.append([message.header.stamp.to_sec(), p.x, p.y, p.z, *velocity_world])
        poses.append((message.header.stamp.to_sec(), np.array([p.x, p.y, p.z]),
                      np.array([q.w, q.x, q.y, q.z])))
    def on_registered(message):
        stamp = message.header.stamp.to_sec()
        registered.append([stamp, message.width*message.height])
        if not poses or stamp-map_stats['last_stamp'] < .4:
            return
        pose_stamp, center, quaternion = poses[-1]
        if abs(stamp-pose_stamp) > .15:
            return
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        origin = center + rotation.reshape(3, 3) @ np.array([0., 0., .2])
        endpoints = np.asarray(list(point_cloud2.read_points(message,
                                field_names=('x', 'y', 'z'), skip_nans=True)), dtype=float)
        if endpoints.size == 0:
            return
        map_stats['integrated_hits'] += mapper.integrate(endpoints, origin, stride=2)
        map_stats['integrated_scans'] += 1
        map_stats['last_stamp'] = stamp
    rospy.Subscriber('/Odometry', Odometry,
                     on_odometry, queue_size=100)
    map_subscriber = rospy.Subscriber('/cloud_registered', PointCloud2,
                                      on_registered, queue_size=100)
    step = .005
    start_stamp = 1000.
    truth = []
    imu_count = cloud_count = 0
    for k in range(int(seconds/step)):
        if rospy.is_shutdown():
            break
        t = k*step
        p, v, a, yaw, yaw_rate = pilot_state(t)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
        data.qpos[:3] = p
        data.qpos[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
        data.qvel[:3] = v
        data.qvel[3:6] = [0., 0., yaw_rate]
        data.time = t
        mujoco.mj_forward(model, data)
        # Specific force uses IMU convention: +9.81 m/s^2 at static hover.
        force = R.T @ (a - np.array([0., 0., -9.81]))
        stamp = rospy.Time.from_sec(start_stamp+t)
        imu_pub.publish(imu_message(stamp, force, [0., 0., yaw_rate]))
        imu_count += 1
        if k and k % 20 == 0:
            scan = lidar.read_body(model, data, uid)
            # The virtual scan is instantaneous at t; header is scan start.
            cloud_pub.publish(cloud_message(scan['points'], rospy.Time.from_sec(start_stamp+t-.1),scan_lines=16,vertical_fov=120.))
            cloud_count += 1
        if k % 20 == 0:
            truth.append([start_stamp+t, *p, yaw, *v])
        time.sleep(step)
    # Allow FAST-LIO to drain its cloud queue before persisting the result.
    time.sleep(.5)
    map_subscriber.unregister()
    snapshot = mapper.snapshot(frame_id='fast_lio_world', source='fast_lio_online_lidar',
                              stamp=map_stats['last_stamp'],
                              metadata={'truth_geometry_access': False,
                                        'registered_cloud_source': '/cloud_registered'})
    snapshot.save(output/'online_slam_map.npz')
    truth = np.asarray(truth)
    odometry = np.asarray(odometry)
    np.savez_compressed(output/'pilot_trace.npz', truth=truth, odometry=odometry,
                        registered_clouds=np.asarray(registered))
    metrics = {}
    if len(odometry) > 1:
        matched = np.column_stack([np.interp(odometry[:, 0], truth[:, 0], truth[:, k]) for k in (1, 2, 3)])
        displacement_error = (odometry[:, 1:4]-odometry[0, 1:4])-(matched-matched[0])
        norm = np.linalg.norm(displacement_error, axis=1)
        metrics = {'displacement_rms_m': float(np.sqrt(np.mean(norm**2))),
                   'displacement_max_m': float(norm.max()),
                   'displacement_final_m': float(norm[-1])}
        truth_velocity=np.column_stack([np.interp(odometry[:,0],truth[:,0],truth[:,k]) for k in (5,6,7)])
        velocity_error=np.linalg.norm(odometry[:,4:7]-truth_velocity,axis=1)
        metrics['velocity_rms_m_s']=float(np.sqrt(np.mean(velocity_error**2)))
        metrics['velocity_final_error_m_s']=float(velocity_error[-1])
    (output/'pilot_summary.json').write_text(json.dumps({
        'seconds_requested': seconds, 'imu_messages': imu_count,
        'lidar_messages': cloud_count, 'odometry_messages': len(odometry),
        'registered_cloud_messages': len(registered),
        'registered_points_total': sum(n for _, n in registered),
        'map_integrated_scans': map_stats['integrated_scans'],
        'map_integrated_hits': map_stats['integrated_hits'],
        'map_occupied_voxels': len(snapshot.occupied_indices),
        'map_observed_voxels': len(snapshot.observed_indices),
        'map_hash': snapshot.map_hash,
        'status': 'odometry_received' if len(odometry) else 'no_odometry',
        'truth_usage': 'evaluation_only', 'point_acquisition': 'instantaneous_at_scan_end',
        **metrics,
    }, indent=2))
    print((output/'pilot_summary.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--output', default=str(Path(__file__).resolve().parent/'runs'/'fast_lio_pilot'))
    args = parser.parse_args()
    run_pilot(args.seconds, args.output)
