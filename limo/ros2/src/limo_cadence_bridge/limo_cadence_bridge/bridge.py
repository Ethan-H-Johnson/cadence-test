"""TCP bridge between a Cadence controller and the LIMO's ROS 2 topics (Jazzy).

Runs unchanged in the Gazebo Harmonic sim and on the rover. The controller may run on the
same machine or connect over the network.

Protocol: one JSON object per line, one reply per request.
  {"op": "read"}               -> {"ranges": [...], "v", "w", "x", "y", "tilt", "t"}
  {"op": "cmd", "v": , "w": }  -> {"ok": true}   (republished at 20 Hz until replaced)
  {"op": "time"}               -> {"t"}          (sim time in Gazebo, wall time on the rover)
  {"op": "viz", "goal": [x, y] | null, "action", "v", "w", "sectors": [8 nearness values]}
                               -> {"ok": true}   (simulation only: draws Gazebo markers)
  {"op": "reset"}              -> {"ok": true}   (simulation only: models back to spawn)

``ranges`` covers the front ``fov_deg`` of the scan (the robot's body blocks the rear),
min-pooled into ``bins`` values from right to left; readings outside the sensor's range
become ``range_max``. ``tilt`` is the IMU's angle from upright in degrees.

Poses are in the arena frame: odometry re-anchored at every reset so the robot's reported pose
matches where the sim put it (the odometry itself keeps counting through a reset). On the
rover, which is never reset, this is the odometry frame: the pose where the driver started.

Safety: the base is commanded to stop if no command arrives for ``watchdog`` seconds, and
when the controller disconnects. Speeds are clamped to ``max_linear`` / ``max_angular``.
"""

import json
import math
import socket
import subprocess
import threading

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan


class Bridge(Node):
    def __init__(self):
        super().__init__("cadence_bridge")
        p = self.declare_parameter
        self.port = p("port", 5555).value
        self.bind = p("bind", "0.0.0.0").value
        self.bins = p("bins", 240).value
        self.fov = math.radians(p("fov_deg", 240.0).value)
        self.watchdog = p("watchdog", 0.5).value
        self.max_v = p("max_linear", 0.5).value
        self.max_w = p("max_angular", 1.5).value
        self.world = p("reset_world", "").value  # Gazebo world name; empty on the rover
        self.model = p("model_name", "limo").value
        self.spawn = (p("spawn_x", 0.0).value, p("spawn_y", 0.0).value, p("spawn_z", 0.02).value)
        self.lock = threading.Lock()
        self.scan = self.odom = None
        self.anchor = (0.0, 0.0, 0.0)  # (dx, dy, dyaw) from the odometry frame to the arena
        self.reanchor = False
        self.markers = None  # gz transport node, created on the first viz request
        self.tilt = 0.0
        self.cmd, self.cmd_time = (0.0, 0.0), self.get_clock().now()
        self.pub = self.create_publisher(Twist, p("cmd_topic", "/cmd_vel").value, 1)
        self.create_subscription(LaserScan, p("scan_topic", "/scan").value, self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, p("odom_topic", "/odom").value, self.on_odom, 10)
        self.create_subscription(Imu, p("imu_topic", "/imu").value, self.on_imu,
                                 qos_profile_sensor_data)
        self.create_timer(0.05, self.publish)

    # ROS side ---------------------------------------------------------------
    def on_scan(self, msg):
        half = self.fov / 2
        front = []
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))  # wrap to [-pi, pi]
            if -half <= a <= half:
                # at or below range_min is "no return" (Gazebo clamps self-hits there, the
                # lidar driver reports 0); treat it like nothing in range
                ok = msg.range_min < r <= msg.range_max and not math.isnan(r)
                front.append((a, r if ok else msg.range_max))
        front.sort()
        n = len(front)
        if n == 0:
            return
        pooled = [min(r for _, r in front[i * n // self.bins:max((i + 1) * n // self.bins, i * n // self.bins + 1)])
                  for i in range(self.bins)]
        with self.lock:
            self.scan = pooled

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            if self.reanchor:  # the robot was just put at the spawn pose
                sx, sy, _ = self.spawn
                dyaw = -yaw  # spawn faces +x
                c, s = math.cos(dyaw), math.sin(dyaw)
                self.anchor = (sx - (c * p.x - s * p.y), sy - (s * p.x + c * p.y), dyaw)
                self.reanchor = False
            dx, dy, dyaw = self.anchor
            c, s = math.cos(dyaw), math.sin(dyaw)
            self.odom = {"x": c * p.x - s * p.y + dx, "y": s * p.x + c * p.y + dy,
                         "yaw": math.atan2(math.sin(yaw + dyaw), math.cos(yaw + dyaw)),
                         "v": msg.twist.twist.linear.x, "w": msg.twist.twist.angular.z}

    def on_imu(self, msg):
        q = msg.orientation
        up_z = 1 - 2 * (q.x * q.x + q.y * q.y)  # body up axis, world z component
        with self.lock:
            self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    def publish(self):
        with self.lock:
            v, w = self.cmd
            if (self.get_clock().now() - self.cmd_time).nanoseconds * 1e-9 > self.watchdog:
                v = w = 0.0
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub.publish(t)

    # controller side ----------------------------------------------------------
    def respond(self, req):  # not "handle": rclpy.Node uses that name
        op = req.get("op")
        if op == "read":
            with self.lock:
                scan, odom, tilt = self.scan, self.odom, self.tilt
            if scan is None or odom is None:
                return {"error": "no scan/odom received yet; check the topic names"}
            if not all(math.isfinite(odom[k]) for k in odom):
                return {"error": "odometry is not finite: %r" % odom}
            out = dict(odom, ranges=scan, tilt=tilt)
            out["t"] = self.get_clock().now().nanoseconds * 1e-9
            return out
        if op == "time":  # cheap clock read for pacing (sim time in Gazebo)
            return {"t": self.get_clock().now().nanoseconds * 1e-9}
        if op == "cmd":
            v = max(-self.max_v, min(self.max_v, float(req["v"])))
            w = max(-self.max_w, min(self.max_w, float(req["w"])))
            with self.lock:
                self.cmd, self.cmd_time = (v, w), self.get_clock().now()
            return {"ok": True}
        if op == "reset":
            return self.reset()
        if op == "viz":
            return self.viz(req)
        return {"error": "unknown op %r" % op}

    def reset(self):
        """Put the robot back at its spawn pose (simulation only).

        Uses the world's set_pose service; Gazebo's WorldControl model reset neither moved
        the robot nor answered reliably (2026-10-15)."""
        if not self.world:
            return {"error": "reset unavailable: not a simulation (reset_world is empty)"}
        with self.lock:
            self.cmd = (0.0, 0.0)
            self.odom = None  # report nothing until odometry from after the reset arrives
            self.reanchor = True
        x, y, z = self.spawn
        done = subprocess.run(
            ["gz", "service", "-s", "/world/%s/set_pose" % self.world,
             "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean", "--timeout", "5000",
             "--req", 'name: "%s", position: {x: %f, y: %f, z: %f}, orientation: {w: 1}'
             % (self.model, x, y, z)],
            capture_output=True, text=True, timeout=10)
        if "true" not in done.stdout:
            return {"error": "gz set_pose failed: %s %s" % (done.stdout.strip(), done.stderr.strip())}
        return {"ok": True}

    # Gazebo markers ------------------------------------------------------------
    # Drawn on the floor (z 0.01-0.03) or above the walls (z 0.5): the lidar scans a single
    # plane ~0.12 m up, so it never sees them.
    def viz(self, req):
        if not self.world:
            return {"error": "viz unavailable: not a simulation"}
        try:
            from gz.msgs10.boolean_pb2 import Boolean
            from gz.msgs10.marker_pb2 import Marker
            from gz.msgs10.marker_v_pb2 import Marker_V
            from gz.transport13 import Node as GzNode
        except ImportError as e:
            return {"error": "gz python bindings missing: %s" % e}
        if self.markers is None:
            self.markers = GzNode()
        with self.lock:
            pose = self.odom
        if pose is None:
            return {"error": "no pose yet"}
        x, y, yaw = pose["x"], pose["y"], pose["yaw"]
        batch = Marker_V()

        def marker(mid, kind, rgba):
            m = batch.marker.add()
            m.ns, m.id, m.action, m.type = "cadence", mid, Marker.ADD_MODIFY, kind
            for c in (m.material.ambient, m.material.diffuse):
                c.r, c.g, c.b, c.a = rgba
            return m

        goal = req.get("goal")
        disc = marker(1, Marker.CYLINDER, (0.1, 0.8, 0.2, 0.6))
        ball = marker(2, Marker.SPHERE, (0.1, 0.9, 0.2, 1.0))
        if goal:
            disc.pose.position.x, disc.pose.position.y, disc.pose.position.z = goal[0], goal[1], 0.005
            disc.scale.x = disc.scale.y = 0.6  # the 0.3 m arrival radius
            disc.scale.z = 0.01
            ball.pose.position.x, ball.pose.position.y, ball.pose.position.z = goal[0], goal[1], 0.5
            ball.scale.x = ball.scale.y = ball.scale.z = 0.15
        else:
            disc.action = ball.action = Marker.DELETE_MARKER

        # the path the chosen command would drive over 1.5 s
        colours = {"forward": (0.2, 0.5, 1.0, 1.0), "veer left": (1.0, 0.6, 0.1, 1.0),
                   "veer right": (1.0, 0.6, 0.1, 1.0), "spin left": (0.8, 0.2, 0.9, 1.0),
                   "spin right": (0.8, 0.2, 0.9, 1.0)}
        path = marker(3, Marker.LINE_STRIP, colours.get(req.get("action"), (1, 1, 1, 1)))
        v, w = float(req.get("v", 0.0)), float(req.get("w", 0.0))
        px, py, pyaw = x, y, yaw
        for _ in range(16):
            pt = path.point.add()
            pt.x, pt.y, pt.z = px, py, 0.03
            pyaw += w * 0.1
            px += max(v, 0.08) * math.cos(pyaw) * 0.1  # spins still show their direction
            py += max(v, 0.08) * math.sin(pyaw) * 0.1

        # the eight sectors the brain sees, left to right across +120..-120 deg
        lx, ly = x + 0.103 * math.cos(yaw), y + 0.103 * math.sin(yaw)  # lidar position
        for i, near in enumerate(req.get("sectors") or []):
            near = max(0.0, min(1.0, float(near)))
            ray = marker(10 + i, Marker.LINE_LIST, (near, 1.0 - near, 0.1, 1.0))
            a = yaw + math.radians(105 - 30 * i)  # sector centre
            length = (1.0 - near) * 1.5 + 0.05
            for d in (0.0, length):
                pt = ray.point.add()
                pt.x, pt.y, pt.z = lx + d * math.cos(a), ly + d * math.sin(a), 0.02

        ok, _ = self.markers.request("/marker_array", batch, Marker_V, Boolean, 200)
        return {"ok": True} if ok else {"error": "marker request failed (is the GUI open?)"}

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind, self.port))
        server.listen(1)
        server.settimeout(1.0)
        self.get_logger().info("cadence bridge listening on %s:%d" % (self.bind, self.port))
        while rclpy.ok():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            self.get_logger().info("controller connected from %s" % addr[0])
            self.client(conn)
            with self.lock:
                self.cmd = (0.0, 0.0)
            self.get_logger().info("controller disconnected; base stopped")

    def client(self, conn):
        f = conn.makefile("rwb")
        try:
            for line in iter(f.readline, b""):
                try:
                    reply = self.respond(json.loads(line))
                except Exception as e:  # report to the controller, keep serving
                    reply = {"error": str(e)}
                f.write((json.dumps(reply) + "\n").encode())
                f.flush()
        except OSError:
            pass
        finally:
            conn.close()


def main():
    rclpy.init()
    node = Bridge()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        node.serve()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
