#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray


class ArmShapeSimNode:
    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.omega = float(rospy.get_param("~omega", 8.0))
        self.zeta = float(rospy.get_param("~zeta", 0.9))
        self.cmd = [0.0, 0.0]
        self.q = [0.0, 0.0]
        self.qd = [0.0, 0.0]
        self.pub = rospy.Publisher("/arm/current_theta_phi", Float32MultiArray, queue_size=10)
        rospy.Subscriber("/arm/theta_phi_cmd", Float32MultiArray, self._cmd_cb, queue_size=10)
        rospy.Subscriber("/continuum_arm/control_angles", Float32MultiArray, self._gui_cb, queue_size=10)

    def _cmd_cb(self, msg):
        if len(msg.data) >= 2:
            self.cmd = [float(msg.data[0]), float(msg.data[1])]

    def _gui_cb(self, msg):
        if len(msg.data) >= 2:
            self.cmd = [float(msg.data[0]), float(msg.data[1])]

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        dt = 1.0 / max(self.rate_hz, 1.0)
        while not rospy.is_shutdown():
            for i in range(2):
                acc = self.omega * self.omega * (self.cmd[i] - self.q[i]) - 2.0 * self.zeta * self.omega * self.qd[i]
                self.qd[i] += acc * dt
                self.q[i] += self.qd[i] * dt
            out = Float32MultiArray()
            out.data = list(self.q)
            self.pub.publish(out)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("arm_shape_sim_node")
    try:
        ArmShapeSimNode().spin()
    except rospy.ROSInterruptException:
        pass
