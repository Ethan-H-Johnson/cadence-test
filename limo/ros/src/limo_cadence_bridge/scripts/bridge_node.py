#!/usr/bin/env python
"""TCP bridge between a Cadence controller and the LIMO's ROS topics.

Runs under ROS Melodic (Python 2.7, the Jetson Nano LIMO) and Noetic (Python 3, Gazebo).
The Cadence process needs Python 3.11+, so it runs separately and connects here.

Protocol: one JSON object per line, one reply per request.
  {"op": "read"}               -> {"ranges": [...], "v", "w", "x", "y", "t"}
  {"op": "cmd", "v": , "w": }  -> {"ok": true}   (republished at 20 Hz until replaced)
  {"op": "reset"}              -> {"ok": true}   (Gazebo only: respawn at the start pose)

Safety: the base is commanded to stop if no command arrives for ``~watchdog`` seconds,
or when the client disconnects.
"""
from __future__ import division, print_function

import json
import math
import socket
import threading

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class Bridge(object):
    def __init__(self):
        self.port = rospy.get_param("~port", 5555)
        self.bind = rospy.get_param("~bind", "0.0.0.0")
        self.bins = rospy.get_param("~bins", 240)
        self.watchdog = rospy.get_param("~watchdog", 0.5)
        self.max_v = rospy.get_param("~max_linear", 0.5)
        self.max_w = rospy.get_param("~max_angular", 1.5)
        self.model = rospy.get_param("~model_name", "limo/")
        self.lock = threading.Lock()
        self.scan = None
        self.odom = None
        self.cmd = (0.0, 0.0)
        self.cmd_time = rospy.Time(0)
        self.pub = rospy.Publisher(rospy.get_param("~cmd_topic", "/cmd_vel"), Twist, queue_size=1)
        rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/odom"), Odometry, self.on_odom, queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self.publish)

    def on_scan(self, msg):
        r = [x if (msg.range_min <= x <= msg.range_max and not math.isnan(x)) else msg.range_max
             for x in msg.ranges]
        # min-pool to a fixed number of bins so sim (720 rays) and robot lidar look alike
        n = len(r)
        pooled = [min(r[i * n // self.bins:max((i + 1) * n // self.bins, i * n // self.bins + 1)])
                  for i in range(self.bins)]
        with self.lock:
            self.scan = pooled

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        # tilt: angle between the body's up axis and the world's, in degrees (180 = upside down)
        up_z = 1 - 2 * (q.x * q.x + q.y * q.y)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
        with self.lock:
            self.odom = {"x": p.x, "y": p.y,
                         "yaw": math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                         "v": msg.twist.twist.linear.x, "w": msg.twist.twist.angular.z, "tilt": tilt}

    def publish(self, _event):
        with self.lock:
            v, w = self.cmd
            if (rospy.Time.now() - self.cmd_time).to_sec() > self.watchdog:
                v, w = 0.0, 0.0
        t = Twist()
        t.linear.x, t.angular.z = v, w
        self.pub.publish(t)

    def handle(self, req):
        op = req.get("op")
        if op == "read":
            with self.lock:
                scan, odom = self.scan, self.odom
            if scan is None or odom is None:
                return {"error": "no scan/odom received yet; check the topic names"}
            if any(math.isnan(odom[k]) or math.isinf(odom[k]) for k in odom):
                return {"error": "odometry is not finite: %r" % odom}
            out = dict(odom)
            out["ranges"] = scan
            out["t"] = rospy.Time.now().to_sec()
            return out
        if op == "cmd":
            v = max(-self.max_v, min(self.max_v, float(req["v"])))
            w = max(-self.max_w, min(self.max_w, float(req["w"])))
            with self.lock:
                self.cmd, self.cmd_time = (v, w), rospy.Time.now()
            return {"ok": True}
        if op == "reset":
            return self.reset()
        return {"error": "unknown op %r" % op}

    def reset(self):
        # reset_world returns every model to its spawn pose with zero velocity; moving the
        # robot with set_model_state at ground height can leave the physics producing NaNs
        try:
            from std_srvs.srv import Empty
            rospy.wait_for_service("/gazebo/reset_world", timeout=2.0)
            with self.lock:
                self.cmd = (0.0, 0.0)
                self.odom = None  # wait for odometry from after the reset
            rospy.ServiceProxy("/gazebo/reset_world", Empty)()
            return {"ok": True}
        except Exception as e:  # no Gazebo: a real robot cannot be respawned
            return {"error": "reset unavailable: %s" % e}

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind, self.port))
        server.listen(1)
        server.settimeout(1.0)
        rospy.loginfo("cadence bridge listening on %s:%d", self.bind, self.port)
        while not rospy.is_shutdown():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            rospy.loginfo("controller connected from %s", addr[0])
            self.client(conn)
            with self.lock:
                self.cmd = (0.0, 0.0)
            rospy.loginfo("controller disconnected; base stopped")

    def client(self, conn):
        f = conn.makefile("rwb")
        try:
            for line in iter(f.readline, b""):
                try:
                    reply = self.handle(json.loads(line.decode("utf-8")))
                except Exception as e:
                    reply = {"error": str(e)}
                f.write((json.dumps(reply) + "\n").encode("utf-8"))
                f.flush()
        except (socket.error, IOError):
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    rospy.init_node("cadence_bridge")
    Bridge().serve()
