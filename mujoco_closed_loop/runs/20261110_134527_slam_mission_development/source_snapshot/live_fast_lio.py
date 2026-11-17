"""ROS1 FAST-LIO sensor link for MuJoCo mission integration.

The only world-frame input is a calibrated initial hover pose. Subsequent
poses and map rays come from FAST-LIO odometry and registered LiDAR returns.
"""
from collections import deque
from pathlib import Path
import hashlib
import json
import sys
import threading

import mujoco
import numpy as np
import rospy
from scipy.spatial import cKDTree
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs import point_cloud2

from fast_lio_bridge import cloud_message, imu_message
from sensors import SurfaceLidar
from robot_volume_prior import RobotVolumePrior
sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src'))
from uav_continuum_whole_body_v2.mapping import IncrementalVoxelMapper, MapSnapshot, voxel_centers


def interpolate_sparse_free_voxels(snapshot, center):
    """Close one-voxel gaps between measured free rays away from hit cells."""
    observed=snapshot.observed_indices
    occupied=snapshot.occupied_indices
    if len(observed)==0 or len(occupied)==0:return snapshot
    resolution=snapshot.resolution
    centers=voxel_centers(observed,resolution)
    nearby=observed[np.all(np.abs(centers-np.asarray(center))<[1.5,1.5,1.],axis=1)]
    if len(nearby)==0:return snapshot
    # A proposal needs at least two distinct face-adjacent free cells. Never
    # infer close to a LiDAR/RGB-D hit; that retains obstacle edges and gaps.
    dtype=np.dtype([('x',np.int64),('y',np.int64),('z',np.int64)])
    occupied_view=np.ascontiguousarray(occupied).view(dtype).ravel()
    nearby=nearby[~np.isin(np.ascontiguousarray(nearby).view(dtype).ravel(),occupied_view)]
    if len(nearby)==0:return snapshot
    faces=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    proposed,counts=np.unique((nearby[:,None,:]+faces[None,:,:]).reshape(-1,3),axis=0,return_counts=True)
    proposed=proposed[counts>=2]
    if len(proposed)==0:return snapshot
    tree=cKDTree(voxel_centers(occupied,resolution))
    distance,_=tree.query(voxel_centers(proposed,resolution),workers=1)
    proposed=proposed[distance>.25]
    proposed=proposed[~np.isin(np.ascontiguousarray(proposed).view(dtype).ravel(),
                              np.ascontiguousarray(observed).view(dtype).ravel())]
    if len(proposed)==0:return snapshot
    original_hash=snapshot.metadata.get('sensor_map_hash_before_free_interpolation',snapshot.map_hash)
    inferred_before=int(snapshot.metadata.get('inferred_free_voxels',0))
    previous_rounds=int(snapshot.metadata.get('free_interpolation_rounds',0))
    result=MapSnapshot(resolution,occupied,np.vstack([observed,proposed]),
        frame_id=snapshot.frame_id,source=snapshot.source,stamp=snapshot.stamp,
        metadata=dict(snapshot.metadata,sensor_map_hash_before_free_interpolation=original_hash,
            free_interpolation='two_face_neighbors_and_0.25m_hit_exclusion',
            free_interpolation_rounds=previous_rounds+1,
            inferred_free_voxels=inferred_before+int(len(proposed))),
        traversed_positions=snapshot.traversed_positions)
    return result

def load_local_sensor_map(path,center,radius):
    """Verify an atomic saved map, then construct only its local observed crop."""
    with np.load(path,allow_pickle=False) as data:
        resolution=float(data['resolution'][0]);occupied=data['occupied_indices']
        observed=data['observed_indices'];frame_id=str(data['frame_id'][0])
        source=str(data['source'][0]);stamp=float(data['stamp'][0])
        metadata=json.loads(str(data['metadata_json'][0]))
        traversed=data['traversed_positions']
    expected=metadata.pop('map_hash',None)
    if expected is None:raise ValueError('RGB-D map has no integrity hash')
    digest=hashlib.sha256()
    digest.update(('resolution=%.12g\n'%resolution).encode('ascii'))
    digest.update(('frame=%s\nsource=%s\n'%(frame_id,source)).encode('utf-8'))
    digest.update(occupied.astype('<i8',copy=False).tobytes(order='C'))
    digest.update(b'\nobserved\n')
    digest.update(observed.astype('<i8',copy=False).tobytes(order='C'))
    if len(traversed):
        digest.update(b'\ntraversed_positions\n')
        digest.update(traversed.astype('<f8',copy=False).tobytes(order='C'))
    canonical={str(k):v for k,v in metadata.items() if k not in ('map_hash','stamp','created_wall_time')}
    digest.update(json.dumps(canonical,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8'))
    if digest.hexdigest()!=expected:raise ValueError('RGB-D map integrity hash mismatch')
    centres=(observed.astype(float)+.5)*resolution
    keep=np.all(np.abs(centres[:,:2]-center[:2])<=radius,axis=1)
    keep&=(centres[:,2]>=center[2]-1.)&(centres[:,2]<=center[2]+1.)
    local=MapSnapshot(resolution,occupied,np.vstack([observed[keep],occupied]),
        frame_id=frame_id,source=source,stamp=stamp,metadata=metadata,
        traversed_positions=traversed)
    return local,expected


def transform(position, quaternion):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.array([quaternion.w, quaternion.x,
                                          quaternion.y, quaternion.z]))
    result = np.eye(4)
    result[:3, :3] = rotation.reshape(3, 3)
    result[:3, 3] = [position.x, position.y, position.z]
    return result


class LiveFastLioLink:
    def __init__(self, initial_pose, output, seed=0, nav_profile='front_housed'):
        self.initial_pose = np.asarray(initial_pose).copy()
        self.output = Path(output)
        self.lock = threading.Lock()
        self.alignment = None
        self.poses = deque(maxlen=50)
        self.pending = deque(maxlen=50)
        self.traversed = []
        self.mapper = IncrementalVoxelMapper(.05, 6.)
        self.lidar = SurfaceLidar(seed=seed,azimuth_count=360,elevation_count=16,vertical_fov=120.)
        self.local_hit_range_m=2.5;self.local_free_range_m=1.5
        self.nav_profile=nav_profile
        self.robot_prior = RobotVolumePrior(profile='grasp_visible',nav_profile=nav_profile,approved_stage1=(nav_profile=='approved_stage1'))
        self.map_last_stamp = -1.
        self.scans = self.imu_messages = self.odom_messages = 0
        self.map_scans = self.map_hits = 0
        self.footprint_prior_cells = 0
        rospy.init_node('mujoco_live_fast_lio', anonymous=True, disable_signals=True)
        self.cloud_pub = rospy.Publisher('/velodyne_points', PointCloud2, queue_size=4)
        self.imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=100)
        self.odom_sub = rospy.Subscriber('/Odometry', Odometry, self._odom, queue_size=100)
        self.cloud_sub = rospy.Subscriber('/cloud_registered', PointCloud2,
                                           self._registered, queue_size=10)

    def _odom(self, message):
        raw = transform(message.pose.pose.position, message.pose.pose.orientation)
        stamp = message.header.stamp.to_sec()-1000.
        with self.lock:
            if self.alignment is None:
                self.alignment = self.initial_pose @ np.linalg.inv(raw)
            pose = self.alignment @ raw
            velocity_body=np.array([message.twist.twist.linear.x,message.twist.twist.linear.y,message.twist.twist.linear.z],dtype=float)
            velocity_world=pose[:3,:3]@velocity_body
            self.poses.append((stamp, pose))
            self.pending.append((stamp, pose, velocity_world))
            if not self.traversed or np.linalg.norm(pose[:3,3]-self.traversed[-1])>.05:
                self.traversed.append(pose[:3,3].copy())
            self.odom_messages += 1

    def _registered(self, message):
        stamp = message.header.stamp.to_sec()-1000.
        with self.lock:
            if not self.poses or stamp-self.map_last_stamp < .4:
                return
            pose_stamp, pose = self.poses[-1]
            if abs(stamp-pose_stamp) > .15:
                return
            alignment = self.alignment.copy()
        points = np.asarray(list(point_cloud2.read_points(message,
                           field_names=('x', 'y', 'z'), skip_nans=True)), dtype=float)
        if points.size == 0:
            return
        endpoints = points @ alignment[:3, :3].T + alignment[:3, 3]
        origin = pose[:3, 3] + pose[:3, :3] @ np.array([0., 0., .2])
        # FAST-LIO still receives the complete 6 m scan.  The rolling V2
        # map integrates first returns only within its 2.5 m local horizon;
        # farther cells remain unknown until the robot approaches them.
        endpoints=endpoints[np.linalg.norm(endpoints-origin,axis=1)<=self.local_hit_range_m]
        if not len(endpoints):return
        with self.lock:
            self.map_hits += self.mapper.integrate(endpoints, origin, stride=2)
            self.map_scans += 1
            self.map_last_stamp = stamp

    def publish_imu(self, stamp, specific_force, angular_velocity):
        self.imu_pub.publish(imu_message(rospy.Time.from_sec(1000.+stamp),
                                         specific_force, angular_velocity))
        self.imu_messages += 1

    def _mark_contact_free_body_footprint(self, center):
        """Current body envelope only; never erase measured occupied cells."""
        res=self.mapper.resolution
        radius=.28+np.sqrt(2)*res/2
        half_height=.145+res/2
        lower=np.floor((center-[radius,radius,half_height])/res).astype(int)
        upper=np.floor((center+[radius,radius,half_height])/res).astype(int)
        added=0
        for ix in range(lower[0],upper[0]+1):
            for iy in range(lower[1],upper[1]+1):
                xy=(np.array([ix,iy])+.5)*res
                if np.linalg.norm(xy-center[:2])>radius:continue
                for iz in range(lower[2],upper[2]+1):
                    z=(iz+.5)*res
                    if abs(z-center[2])>half_height:continue
                    key=(ix,iy,iz)
                    if key not in self.mapper.occupied and key not in self.mapper.observed:
                        self.mapper.observed.add(key);added+=1
        self.footprint_prior_cells+=added

    def publish_scan(self, model, data, body_id, bending=None, contact_free=True):
        scan = self.lidar.read_body(model, data, body_id)
        if not len(scan['points']):
            return
        self.cloud_pub.publish(cloud_message(scan['points'],
                         rospy.Time.from_sec(1000.+float(data.time)-.1),scan_lines=16,vertical_fov=120.))
        self.scans += 1
        # No-return rays are explicit sensor evidence of free space up to the
        # range limit. Decimate only for CPU cost; never extend beyond range.
        latest=self.latest_pose()
        if latest is not None and abs(data.time-latest[0])<.2:
            _,pose=latest
            origin=pose[:3,3]+pose[:3,:3]@np.array([0.,0.,.2])
            far_free=None
            if self.scans%3==0:
                raw=scan['points'];distances=np.linalg.norm(raw,axis=1)
                far=raw[distances>self.local_hit_range_m]
                if len(far):
                    far_free=(far*(self.local_free_range_m/np.linalg.norm(far,axis=1))[:,None])@pose[:3,:3].T+origin
            with self.lock:
                if self.scans%3==0:
                    if far_free is not None:self.mapper.integrate_free_rays(far_free,origin,stride=1)
                    # A no-return beam proves free space, but the local map
                    # needs only its nearest 1.5 m.  Never extend a ray.
                    free=(scan['free_endpoints']*(self.local_free_range_m/self.lidar.maximum_range))@pose[:3,:3].T+origin
                    self.mapper.integrate_free_rays(free,origin,stride=4)
                if bending is not None:
                    self.robot_prior.integrate(self.mapper,pose,bending)
                if contact_free:
                    self._mark_contact_free_body_footprint(pose[:3,3])

    def drain_poses(self):
        with self.lock:
            result = list(self.pending)
            self.pending.clear()
        return result

    def latest_pose(self):
        with self.lock:
            return None if not self.poses else (self.poses[-1][0],self.poses[-1][1].copy())

    def snapshot(self, path, rgbd_path=None, local_center=None, local_radius=1.):
        with self.lock:
            # Copy mutable ROS callback state quickly; voxel sorting, merging
            # and compression run outside the callback lock.
            occupied=self.mapper.occupied.copy()
            observed=self.mapper.observed.copy()
            traversed=np.asarray(self.traversed,dtype=float).reshape(-1,3).copy()
            stamp=self.map_last_stamp
            prior_cells=self.footprint_prior_cells
            resolution=self.mapper.resolution
        occupied_indices=np.asarray(list(occupied),dtype=np.int64).reshape(-1,3)
        observed_indices=np.asarray(list(observed),dtype=np.int64).reshape(-1,3)
        if local_center is not None and len(observed_indices):
            center=np.asarray(local_center,dtype=float)
            centres=(observed_indices.astype(float)+.5)*resolution
            keep=np.all(np.abs(centres[:,:2]-center[:2])<=local_radius,axis=1)
            keep&=(centres[:,2]>=center[2]-1.)&(centres[:,2]<=center[2]+1.)
            observed_indices=observed_indices[keep]
        snapshot = MapSnapshot(resolution,
            occupied_indices,np.vstack([observed_indices,occupied_indices]),
            frame_id='world',source='fast_lio_online_lidar',stamp=stamp,
            metadata={'truth_geometry_access':False,
                'registered_cloud_source':'/cloud_registered',
                'navigation_camera_profile':self.nav_profile,
                'robot_visual_profile':'approved_stage1' if self.nav_profile=='approved_stage1' else 'development_base',
                'local_hit_range_m':self.local_hit_range_m,
                'local_no_return_free_range_m':self.local_free_range_m,
                'far_return_free_prefix_m':self.local_free_range_m,
                'contact_free_body_footprint_prior_cells':prior_cells,
                'footprint_prior_erases_occupied':False},
            traversed_positions=traversed)
        def local_observed(source):
            if local_center is None:return source.observed_indices
            center=np.asarray(local_center,dtype=float)
            points=source.observed_points
            keep=np.all(np.abs(points[:,:2]-center[:2])<=local_radius,axis=1)
            keep&=(points[:,2]>=center[2]-1.)&(points[:,2]<=center[2]+1.)
            return source.observed_indices[keep]
        if rgbd_path is not None and Path(rgbd_path).exists():
            if local_center is None:rgbd=MapSnapshot.load(rgbd_path);rgbd_full_hash=rgbd.map_hash
            else:rgbd,rgbd_full_hash=load_local_sensor_map(rgbd_path,np.asarray(local_center,dtype=float),local_radius)
            if rgbd.frame_id!='world' or abs(rgbd.resolution-snapshot.resolution)>1e-12:
                raise ValueError('FAST-LIO and RGB-D maps use different frames or grids')
            snapshot=MapSnapshot(snapshot.resolution,
                np.vstack([snapshot.occupied_indices,rgbd.occupied_indices]),
                np.vstack([local_observed(snapshot),local_observed(rgbd),
                           snapshot.occupied_indices,rgbd.occupied_indices]),
                frame_id='world',source='fast_lio_lidar_rgbd_online',
                stamp=max(snapshot.stamp,rgbd.stamp),
                metadata={'truth_geometry_access':False,'lidar_map_hash':snapshot.map_hash,
                          'rgbd_map_hash':rgbd_full_hash,'registration':'FAST_LIO_and_measured_strain',
                          'robot_visual_profile':'approved_stage1' if self.nav_profile=='approved_stage1' else 'development_base',
                          'local_observed_center_m':None if local_center is None else np.asarray(local_center).tolist(),
                          'local_observed_radius_m':local_radius if local_center is not None else None},
                traversed_positions=np.vstack([snapshot.traversed_positions,rgbd.traversed_positions]))
        elif local_center is not None:
            snapshot=MapSnapshot(snapshot.resolution,snapshot.occupied_indices,
                np.vstack([local_observed(snapshot),snapshot.occupied_indices]),
                frame_id='world',source='fast_lio_online_lidar_local',stamp=snapshot.stamp,
                metadata=dict(snapshot.metadata,local_observed_center_m=np.asarray(local_center).tolist(),local_observed_radius_m=local_radius),
                traversed_positions=snapshot.traversed_positions)
        raw_path=Path(path).with_name(Path(path).stem+'_sensor_raw.npz')
        snapshot.save(raw_path)
        center=np.asarray(local_center if local_center is not None else
                          (traversed[-1] if len(traversed) else [0.,0.,0.]))
        for _ in range(2):
            snapshot=interpolate_sparse_free_voxels(snapshot,center)
        snapshot.save(path)
        return snapshot

    def close(self):
        self.odom_sub.unregister()
        self.cloud_sub.unregister()
        self.snapshot(self.output/'online_slam_map.npz',self.output/'rgbd_online_map.npz')
        return dict(imu_messages=self.imu_messages,lidar_messages=self.scans,
                    odom_messages=self.odom_messages,map_scans=self.map_scans,
                    map_hits=self.map_hits,footprint_prior_cells=self.footprint_prior_cells)
