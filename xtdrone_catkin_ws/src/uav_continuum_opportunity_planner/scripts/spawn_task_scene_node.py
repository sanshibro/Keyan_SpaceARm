#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose

from uav_continuum_opportunity_planner.common import boxes_from_items, load_target, load_task_boxes_raw


def pose_from_xyz(x, y, z):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = 1.0
    return pose


def box_sdf(name, size, rgba):
    return """<sdf version='1.6'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
      <visual name='visual'>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <ambient>{rgba}</ambient>
          <diffuse>{rgba}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(name=name, sx=size[0], sy=size[1], sz=size[2], rgba=rgba)


def sphere_sdf(name, radius, rgba):
    return """<sdf version='1.6'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
      </collision>
      <visual name='visual'>
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material>
          <ambient>{rgba}</ambient>
          <diffuse>{rgba}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(name=name, radius=radius, rgba=rgba)


class SpawnTaskSceneNode:
    def __init__(self):
        self.boxes = boxes_from_items(load_task_boxes_raw())
        self.target = load_target()
        self.prefix = rospy.get_param("~model_prefix", "opp_")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    def _replace_model(self, name, sdf, pose):
        try:
            self.delete(name)
        except Exception:
            pass
        try:
            res = self.spawn(name, sdf, "", pose, "world")
            if not res.success:
                rospy.logwarn("spawn_task_scene_node: failed to spawn %s: %s", name, res.status_message)
        except Exception as exc:
            rospy.logwarn("spawn_task_scene_node: exception spawning %s: %s", name, exc)

    def run_once(self):
        for box in self.boxes:
            name = self.prefix + box.name
            sdf = box_sdf(name, box.size, "0.70 0.12 0.10 0.85")
            self._replace_model(name, sdf, pose_from_xyz(*box.center))

        target_name = self.prefix + self.target["name"]
        target_pose = self.target["pose"]
        sdf = sphere_sdf(target_name, float(rospy.get_param("~target_radius", 0.10)), "1.0 0.0 1.0 1.0")
        self._replace_model(target_name, sdf, pose_from_xyz(target_pose[0], target_pose[1], target_pose[2]))
        rospy.loginfo("spawn_task_scene_node: spawned %d task boxes and target %s", len(self.boxes), target_name)


if __name__ == "__main__":
    rospy.init_node("spawn_task_scene_node")
    SpawnTaskSceneNode().run_once()
    rospy.spin()
