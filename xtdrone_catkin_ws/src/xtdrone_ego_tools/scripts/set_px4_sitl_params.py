#!/usr/bin/env python3
import time

import rospy
from mavros_msgs.msg import ParamValue
from mavros_msgs.srv import ParamGet, ParamPull, ParamSet


def set_value_with_retry(proxy, param_id, value, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            response = proxy(param_id=param_id, value=value)
            if response.success:
                return
            last_error = "success=false"
        except rospy.ServiceException as error:
            last_error = str(error)
        rospy.sleep(0.25)
    raise RuntimeError("failed to set {}: {}".format(param_id, last_error))


def set_integer_param(proxy, param_id, integer_value):
    value = ParamValue(integer=integer_value, real=0.0)
    set_value_with_retry(proxy, param_id, value)
    rospy.loginfo("set %s = %s", param_id, integer_value)


def set_real_param(proxy, param_id, real_value):
    value = ParamValue(integer=0, real=real_value)
    set_value_with_retry(proxy, param_id, value)
    rospy.loginfo("set %s = %.3f", param_id, real_value)


if __name__ == "__main__":
    rospy.init_node("set_px4_sitl_params")
    namespace = rospy.get_param("~namespace", "/iris_0")
    profile = str(rospy.get_param("~profile", "takeoff")).strip().lower()
    if profile not in ("takeoff", "tracking"):
        raise RuntimeError("unknown PX4 SITL profile: {}".format(profile))
    service_name = namespace.rstrip("/") + "/mavros/param/set"
    get_service_name = namespace.rstrip("/") + "/mavros/param/get"
    pull_service_name = namespace.rstrip("/") + "/mavros/param/pull"
    rospy.wait_for_service(service_name, timeout=15.0)
    rospy.wait_for_service(get_service_name, timeout=15.0)
    rospy.wait_for_service(pull_service_name, timeout=15.0)
    param_set = rospy.ServiceProxy(service_name, ParamSet)
    param_get = rospy.ServiceProxy(get_service_name, ParamGet)
    param_pull = rospy.ServiceProxy(pull_service_name, ParamPull)

    # These estimator selections require reboot and therefore belong to the
    # iris_vision_hgt airframe, not to a post-boot tuning script.  Fail fast if
    # somebody overrides the launch back to the GPS/barometer iris profile.
    required_integer_params = {"EKF2_AID_MASK": 24, "EKF2_HGT_MODE": 3}
    deadline = time.monotonic() + max(
        1.0, float(rospy.get_param("~param_ready_timeout_s", 45.0))
    )
    received_count = 0
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            pull_response = param_pull(force_pull=False)
            if pull_response.success and pull_response.param_received > 0:
                received_count = int(pull_response.param_received)
                break
        except rospy.ServiceException:
            pass
        rospy.sleep(0.25)
    else:
        raise RuntimeError("MAVROS full parameter pull was not ready before timeout")
    rospy.loginfo("MAVROS parameter cache ready: %d parameters", received_count)

    actual_values = {}
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        actual_values = {}
        cache_ready = True
        for param_id in required_integer_params:
            try:
                response = param_get(param_id=param_id)
            except rospy.ServiceException:
                cache_ready = False
                break
            if not response.success:
                cache_ready = False
                break
            actual_values[param_id] = response.value.integer
        if cache_ready:
            break
        rospy.sleep(0.25)
    else:
        raise RuntimeError(
            "MAVROS parameter cache was not ready before timeout"
        )
    for param_id, expected in required_integer_params.items():
        actual = actual_values[param_id]
        if actual != expected:
            raise RuntimeError(
                "{}={} but V2 requires {}; launch iris_vision_hgt and restart PX4".format(
                    param_id, actual, expected
                )
            )

    # Disable RC-loss behavior for keyboard/offboard-only SITL.
    set_integer_param(param_set, "COM_RC_IN_MODE", 4)
    set_integer_param(param_set, "NAV_RCL_ACT", 0)
    set_integer_param(param_set, "COM_RCL_EXCEPT", 4)

    # Match the low-speed whole-body experiment envelope. PX4's generic SITL
    # defaults (12 m/s and 45 deg) make a ground-position transient needlessly
    # violent and are inconsistent with the planner's 0.22 m/s limit.
    set_real_param(param_set, "MPC_XY_VEL_MAX", 1.0)
    set_real_param(param_set, "MPC_Z_VEL_MAX_UP", 0.5)
    set_real_param(param_set, "MPC_Z_VEL_MAX_DN", 0.4)
    set_real_param(param_set, "MPC_TILTMAX_AIR", 20.0)
    set_real_param(param_set, "MPC_TILTMAX_LND", 10.0)
    # Restore the repeatable velocity-loop baseline explicitly.  Calibration
    # scripts vary D online; no trial value may leak into a later flight.
    set_real_param(
        param_set, "MPC_XY_VEL_P_ACC", 1.8 if profile == "takeoff" else 3.0
    )
    set_real_param(param_set, "MPC_XY_VEL_I_ACC", 0.4)
    set_real_param(param_set, "MPC_XY_VEL_D_ACC", 0.2)
    set_real_param(param_set, "MPC_Z_VEL_P_ACC", 4.0)
    set_real_param(param_set, "MPC_Z_VEL_I_ACC", 2.0)
    set_real_param(param_set, "MPC_Z_VEL_D_ACC", 0.0)
    if profile == "takeoff":
        # Gentle enough to prevent the spawn/ground transient from turning
        # into horizontal runaway.
        set_real_param(param_set, "MPC_ACC_HOR", 1.0)
        set_real_param(param_set, "MPC_ACC_UP_MAX", 1.0)
        set_real_param(param_set, "MPC_ACC_DOWN_MAX", 0.8)
        set_real_param(param_set, "MPC_XY_P", 0.95)
        set_real_param(param_set, "MPC_Z_P", 1.0)
    else:
        # The certified trajectory itself remains limited to 1 m/s^2.  The
        # controller needs additional authority beyond that command envelope
        # to reject estimator drift and track narrow-passage height changes.
        set_real_param(param_set, "MPC_ACC_HOR", 2.0)
        set_real_param(param_set, "MPC_ACC_UP_MAX", 2.0)
        set_real_param(param_set, "MPC_ACC_DOWN_MAX", 1.5)
        # Open-space A/B tests retained XYZ velocity feed-forward and found a
        # moderate vertical position gain less error-prone than the 1.0
        # default or an aggressive 2.0 setting.  Keep the horizontal gain at
        # the PX4 default; increasing both axes degraded lateral tracking.
        set_real_param(param_set, "MPC_XY_P", 0.95)
        set_real_param(param_set, "MPC_Z_P", 1.5)
    rospy.loginfo("PX4 SITL profile applied: %s", profile)
