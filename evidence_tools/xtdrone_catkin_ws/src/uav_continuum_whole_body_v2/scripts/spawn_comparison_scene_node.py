#!/usr/bin/env python3
"""Spawn hidden experiment truth into Gazebo; never publishes it to planners."""

from pathlib import Path
import math

import rospy
import yaml
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose
from std_msgs.msg import String


def box_sdf(name, size):
    return """<sdf version='1.6'><model name='{name}'><static>true</static>
<link name='link'><collision name='collision'><geometry><box><size>{size}</size></box></geometry></collision>
<visual name='visual'><geometry><box><size>{size}</size></box></geometry>
<material><ambient>0.58 0.24 0.12 1</ambient><diffuse>0.72 0.34 0.16 1</diffuse></material>
</visual></link></model></sdf>""".format(
        name=name, size=" ".join(str(float(value)) for value in size)
    )


class SpawnComparisonSceneNode:
    def __init__(self):
        oracle_path = Path(rospy.get_param("~oracle_file"))
        if not oracle_path.is_file():
            raise FileNotFoundError("oracle file does not exist: %s" % oracle_path)
        with oracle_path.open("r", encoding="utf-8") as handle:
            oracle = yaml.safe_load(handle) or {}
        if oracle.get("oracle_only") is not True:
            raise ValueError("comparison scene truth must carry oracle_only=true")
        self.boxes = list(oracle.get("boxes", []))
        self.prefix = str(rospy.get_param("~model_prefix", "paper_scene_"))
        self.status = rospy.Publisher(
            "/experiment/simulator_scene_status", String, queue_size=1, latch=True
        )
        rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=30.0)
        self.spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    def run(self):
        spawned = []
        for index, item in enumerate(self.boxes):
            name = self.prefix + str(item.get("name", "box_%03d" % index))
            try:
                self.delete(name)
            except Exception:
                pass
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [
                float(value) for value in item["center"]
            ]
            yaw = float(item.get("yaw", math.radians(item.get("yaw_deg", 0.0))))
            pose.orientation.z = math.sin(0.5 * yaw)
            pose.orientation.w = math.cos(0.5 * yaw)
            response = self.spawn(
                name, box_sdf(name, item["size"]), "", pose, "world"
            )
            if not response.success:
                raise RuntimeError("failed to spawn %s: %s" % (
                    name, response.status_message,
                ))
            spawned.append(name)
        self.status.publish(String(data=(
            "READY oracle_hidden=true boxes=%d" % len(spawned)
        )))
        rospy.loginfo(
            "comparison simulator setup spawned %d hidden-truth boxes", len(spawned)
        )


if __name__ == "__main__":
    rospy.init_node("spawn_comparison_scene_node")
    node = SpawnComparisonSceneNode()
    node.run()
    rospy.spin()
