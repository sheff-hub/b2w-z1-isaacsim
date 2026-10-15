"""ROS 2 link between the simulator loop and Nav2 (the b2w-nav2:jazzy container, nav2/).

Import AFTER the AppLauncher has started. The Isaac container already sets ROS_DISTRO=jazzy,
RMW_IMPLEMENTATION=rmw_fastrtps_cpp, the UDP-only FastDDS profile and the bundled Jazzy
libraries (isaac_env.sh); enabling isaacsim.ros2.bridge puts its rclpy on sys.path.

What the simulator provides, as a real B2W would through its drivers:
  /clock               sim time (Nav2 runs with use_sim_time)
  /tf                  odom -> base_link (ground truth here; odometry + localisation on a robot)
  /tf_static           map -> odom (identity: the simulator IS the map frame)
  /odom                nav_msgs/Odometry, base frame twist
  /scan                sensor_msgs/LaserScan, 360 deg: the NEAREST obstacle over three ray heights
                       (0.10 / 0.35 / 0.57 m) -- one plane at lidar height would see only the
                       table's legs and plan straight under its top
and what it takes back:
  /cmd_vel             geometry_msgs/Twist (vx, vy, wz) -> the ETH base controller
plus Nav2 action clients (NavigateToPose, ComputePathToPose) and, for recording, the latest
/plan (global path) and the controller's local trajectory.

The static map (map_server) is built once from the simulator's colliders: build_map().
"""

import math
import os
import time

import numpy as np

# ROS 2 is loaded lazily (_load_ros, from Nav2Link). Start the app with
# --kit_args "--enable isaacsim.ros2.bridge": enabling the bridge at run time reloads omni.replicator,
# which then cannot rebuild its graph (/Replicator/SDGPipeline), so Isaac Lab's env.seed() and
# reset(seed=...) fail for every seed but 0 (replicator's default, which skips that step).
rclpy = ActionClient = None


def _load_ros():
    global rclpy, ActionClient, DurabilityPolicy, QoSProfile, ReliabilityPolicy, Time, PoseStamped, \
        TransformStamped, Twist, ComputePathToPose, NavigateToPose, Odometry, Path, Clock, LaserScan, Trigger, TFMessage, \
        PointCloud2, PointField, DockRobot, NavigateThroughPoses, FollowPath
    if rclpy is not None:
        return
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")
    import rclpy as _rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
    from nav2_msgs.action import ComputePathToPose, DockRobot, FollowPath, NavigateThroughPoses, NavigateToPose
    from nav_msgs.msg import Odometry, Path
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan, PointCloud2, PointField
    from std_srvs.srv import Trigger
    from tf2_msgs.msg import TFMessage
    rclpy = _rclpy


from omni.physx import get_physx_scene_query_interface
import carb

SCAN_HEIGHTS = (0.10, 0.35, 0.57)  # m above the floor; the table top is 0.56-0.60 m
SCAN_RAYS = 360
SCAN_RANGE = (0.05, 8.0)
SCAN_PERIOD = 0.1  # s of sim time
SKIP = ("/Robot",)  # the robot's own colliders
SELF_COLUMN = ((-0.75, 0.85), (-0.50, 0.50), 1.40)  # OS0 self filter, base_link (floor, yaw) frame: x, y, top z
SELF_RADIUS = 0.22  # m round each robot link / carried box (self_points)


def enable_queries(env_cfg):
    """Isaac Lab turns PhysX scene queries off by default (PhysxCfg.enable_scene_query_support):
    every ray cast and overlap then silently returns nothing. Call on the env cfg before gym.make."""
    env_cfg.sim.physx.enable_scene_query_support = True


def _stamp(t):
    s = int(t)
    return Time(sec=s, nanosec=int(round((t - s) * 1e9)))


def _yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class Scanner:
    """A 2D 'lidar' from PhysX ray casts: per bearing, the nearest hit over SCAN_HEIGHTS that is
    not the robot itself. Rays start at the base centre and are level in the world."""

    def __init__(self, n=SCAN_RAYS, heights=SCAN_HEIGHTS, rmax=SCAN_RANGE[1]):
        self.q = get_physx_scene_query_interface()
        self.n, self.heights, self.rmax = n, heights, rmax
        self.angles = -math.pi + np.arange(n) * (2 * math.pi / n)

    def _first(self, origin, d):
        best = [self.rmax + 1.0]

        def report(hit):
            path = hit.collision if hasattr(hit, "collision") else hit.rigid_body
            if not any(s in path for s in SKIP) and hit.distance < best[0]:
                best[0] = hit.distance
            return True

        self.q.raycast_all(carb.Float3(*origin), carb.Float3(*d), self.rmax, report)
        return best[0]

    def scan(self, x, y, yaw):
        """Ranges (m, inf if nothing) for bearings self.angles in the BASE frame."""
        out = np.full(self.n, np.inf, dtype=np.float32)
        for i, a in enumerate(self.angles):
            c, s = math.cos(yaw + a), math.sin(yaw + a)
            r = min(self._first((x, y, h), (c, s, 0.0)) for h in self.heights)
            if r <= self.rmax:
                out[i] = r
        return out


def build_map(path_prefix, x_range, y_range, res=0.05, z_band=(0.05, 0.9), skip=SKIP):
    """Occupancy grid from the simulator's colliders: a cell is occupied if anything other than
    the robot overlaps the box (cell x cell x z_band) above it. Writes <prefix>.pgm/.yaml for
    nav2 map_server and returns the grid (rows = y, row 0 at y_range[0])."""
    q = get_physx_scene_query_interface()
    nx, ny = int(round((x_range[1] - x_range[0]) / res)), int(round((y_range[1] - y_range[0]) / res))
    # 2 mm smaller than the cell: a box that only TOUCHES an obstacle counts as overlapping, which grew
    # every obstacle by a whole cell (the table edge moved 5 cm out and put the pick point in collision)
    half = carb.Float3(res / 2 - 0.002, res / 2 - 0.002, (z_band[1] - z_band[0]) / 2)
    zc = (z_band[0] + z_band[1]) / 2
    rot = carb.Float4(0.0, 0.0, 0.0, 1.0)
    occ = np.zeros((ny, nx), dtype=bool)
    hit = [False]

    def report(h):
        if not any(s in h.collision for s in skip):
            hit[0] = True
            return False  # stop at the first real hit
        return True

    t0 = time.time()
    for j in range(ny):
        y = y_range[0] + (j + 0.5) * res
        for i in range(nx):
            hit[0] = False
            q.overlap_box(half, carb.Float3(x_range[0] + (i + 0.5) * res, y, zc), rot, report, False)
            occ[j, i] = hit[0]
    img = np.where(occ, 0, 254).astype(np.uint8)[::-1]  # PGM row 0 = top = max y
    with open(path_prefix + ".pgm", "wb") as f:
        f.write(f"P5\n{nx} {ny}\n255\n".encode())
        f.write(img.tobytes())
    with open(path_prefix + ".yaml", "w") as f:
        f.write(f"image: {os.path.basename(path_prefix)}.pgm\nmode: trinary\nresolution: {res}\n"
                f"origin: [{x_range[0]}, {y_range[0]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    print(f"[nav2_link] map {nx}x{ny} at {res} m, {occ.mean() * 100:.1f}% occupied, {time.time() - t0:.1f} s "
          f"-> {path_prefix}.pgm", flush=True)
    return occ


def _mat_to_quat(R):
    """3x3 (column-vector convention) -> (x, y, z, w)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


class Nav2Link:
    """Publishes the robot's state, receives /cmd_vel, and drives Nav2's actions. Call tick() once
    per simulation step (sim time t, base pose, base-frame velocities).

    sensor: "os0" -- the robot's own RTX lidar as a PointCloud2 on /points (frame os0, with a
    base_link -> os0 transform every step: the body's height and pitch change); "scan" -- the
    3-height ray-cast /scan only; "both"."""

    def __init__(self, name="b2w_sim", sensor="os0", lidar_link_pose=None, self_points=None):
        _load_ros()
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(name)
        n = self.node
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_clock = n.create_publisher(Clock, "/clock", 10)
        self.pub_tf = n.create_publisher(TFMessage, "/tf", 50)
        self.pub_tf_static = n.create_publisher(TFMessage, "/tf_static", latched)
        self.pub_odom = n.create_publisher(Odometry, "/odom", 20)
        self.pub_scan = n.create_publisher(LaserScan, "/scan", 10)
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_time = -1.0
        self.t = 0.0
        n.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.plan, self.local_plan = None, None
        n.create_subscription(Path, "/plan", lambda m: setattr(self, "plan", m), 5)
        n.create_subscription(Path, "/local_plan", lambda m: setattr(self, "local_plan", m), 5)
        self.nav_client = ActionClient(n, NavigateToPose, "navigate_to_pose")
        self.path_client = ActionClient(n, ComputePathToPose, "compute_path_to_pose")
        self.active_srv = n.create_client(Trigger, "/lifecycle_manager_navigation/is_active")
        self.sensor = sensor
        self.scanner = Scanner() if sensor in ("scan", "both") else None
        self.os0 = None
        if sensor in ("os0", "both"):
            import os0_lidar
            self.os0 = os0_lidar.OS0(lidar_link_pose)
            self.pub_points = n.create_publisher(PointCloud2, "/points", 5)
        # self_points() -> (N, 3) world points of the robot itself (its links, a carried box): lidar returns
        # within SELF_RADIUS of them are the robot, not obstacles
        self.self_points = self_points
        self.dock_client = ActionClient(n, DockRobot, "dock_robot")
        self.through_client = ActionClient(n, NavigateThroughPoses, "navigate_through_poses")
        self.follow_client = ActionClient(n, FollowPath, "follow_path")  # the controller server, no planner
        self._stream = []  # futures of the goals sent by stream_* (the newest replaces the previous one)
        self.last_scan = -1.0
        self.goal_handle, self.result = None, None
        self._static_sent = 0

    # ---------------------------------------------------------------- state out, command in
    def _on_cmd(self, m):
        self.cmd = (m.linear.x, m.linear.y, m.angular.z)
        self.cmd_time = self.t

    def tick(self, t, x, y, yaw, v_b, w_z):
        """t sim time (s); (x, y, yaw) base pose in the map; v_b (vx, vy) and w_z in the base frame."""
        self.t = t
        st = _stamp(t)
        self.pub_clock.publish(Clock(clock=st))
        if self._static_sent < 3 or int(t * 10) % 20 == 0:  # latched, but re-send for late joiners
            ms = TransformStamped()
            ms.header.stamp, ms.header.frame_id, ms.child_frame_id = st, "map", "odom"
            ms.transform.rotation.w = 1.0
            self.pub_tf_static.publish(TFMessage(transforms=[ms]))
            self._static_sent += 1
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = st, "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y = float(x), float(y)
        qx, qy, qz, qw = _yaw_quat(yaw)
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.pub_tf.publish(TFMessage(transforms=[tf]))
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = st, "odom", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = float(x), float(y)
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = qz, qw
        od.twist.twist.linear.x, od.twist.twist.linear.y = float(v_b[0]), float(v_b[1])
        od.twist.twist.angular.z = float(w_z)
        self.pub_odom.publish(od)
        if self.os0 is not None:
            self.os0.grab()
        if t - self.last_scan >= SCAN_PERIOD - 1e-6 and self.os0 is not None:
            pts, T = self.os0.take()
            # the robot's own body: drop points inside its column (base_link on the floor, yaw only). With the
            # arm stowed the lidar sees none of it, but carrying, the raised arm and the box are in view and Nav2
            # found an obstacle inside the robot (the delivery timed out)
            if len(pts):
                pw_ = pts @ T[:3, :3] + T[3, :3]
                c_, s2 = math.cos(yaw), math.sin(yaw)
                dx_, dy_ = pw_[:, 0] - x, pw_[:, 1] - y
                bx, by = c_ * dx_ + s2 * dy_, -s2 * dx_ + c_ * dy_
                (x0_, x1_), (y0_, y1_), z1_ = SELF_COLUMN
                keep = ~((bx > x0_) & (bx < x1_) & (by > y0_) & (by < y1_) & (pw_[:, 2] < z1_))
                if self.self_points is not None:  # the arm and a carried box can reach outside that column
                    sp = np.asarray(self.self_points(), dtype=np.float32).reshape(-1, 3)
                    near = np.zeros(len(pw_), dtype=bool)
                    for q in sp:
                        near |= np.sum((pw_ - q) ** 2, axis=1) < SELF_RADIUS ** 2
                    keep &= ~near
                pts = pts[keep]
            # base_link (on the floor, yaw only) -> os0, from the sensor's world pose
            Rw = T[:3, :3].T  # USD row-vector -> column-vector rotation (sensor axes in the world)
            pw = T[3, :3]
            c, s_ = math.cos(yaw), math.sin(yaw)
            Rb = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])  # world -> base_link
            rel_p = Rb @ (pw - np.array([x, y, 0.0]))
            tf_l = TransformStamped()
            tf_l.header.stamp, tf_l.header.frame_id, tf_l.child_frame_id = st, "base_link", "os0"
            tf_l.transform.translation.x, tf_l.transform.translation.y, tf_l.transform.translation.z = map(float, rel_p)
            qx_, qy_, qz_, qw_ = _mat_to_quat(Rb @ Rw)
            tf_l.transform.rotation.x, tf_l.transform.rotation.y = float(qx_), float(qy_)
            tf_l.transform.rotation.z, tf_l.transform.rotation.w = float(qz_), float(qw_)
            self.pub_tf.publish(TFMessage(transforms=[tf_l]))
            pc = PointCloud2()
            pc.header.stamp, pc.header.frame_id = st, "os0"
            pc.height, pc.width = 1, int(len(pts))
            pc.fields = [PointField(name=nm, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, nm in enumerate("xyz")]
            pc.is_bigendian, pc.point_step, pc.row_step, pc.is_dense = False, 12, 12 * int(len(pts)), True
            pc.data = np.ascontiguousarray(pts, dtype=np.float32).tobytes()
            self.pub_points.publish(pc)
        if t - self.last_scan >= SCAN_PERIOD - 1e-6:
            self.last_scan = t
        if self.scanner is not None and self.last_scan == t:
            r = self.scanner.scan(x, y, yaw)
            sc = LaserScan()
            sc.header.stamp, sc.header.frame_id = st, "base_link"
            sc.angle_min, sc.angle_increment = float(self.scanner.angles[0]), 2 * math.pi / self.scanner.n
            sc.angle_max = float(self.scanner.angles[-1])
            sc.scan_time, sc.range_min, sc.range_max = SCAN_PERIOD, SCAN_RANGE[0], SCAN_RANGE[1]
            sc.ranges = [float(v) for v in r]
            self.pub_scan.publish(sc)
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def plan_points(self):
        """Nav2's latest global plan (/plan) as [(x, y), ...] in the map, or None."""
        if self.plan is None or not self.plan.poses:
            return None
        return [(p.pose.position.x, p.pose.position.y) for p in self.plan.poses]

    def clear_plan(self):
        self.plan = None

    def command(self, stale_after=0.5):
        """Latest /cmd_vel, zeroed if Nav2 has not sent one for stale_after s of sim time."""
        return self.cmd if self.t - self.cmd_time <= stale_after else (0.0, 0.0, 0.0)

    def spin(self, n=1):
        for _ in range(n):
            rclpy.spin_once(self.node, timeout_sec=0.0)

    # ---------------------------------------------------------------- Nav2 actions
    def _pose(self, x, y, yaw):
        p = PoseStamped()
        p.header.frame_id, p.header.stamp = "map", _stamp(self.t)
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        _, _, p.pose.orientation.z, p.pose.orientation.w = _yaw_quat(yaw)
        return p

    def wait_ready(self, timeout=60.0, tick=None):
        """Wait until Nav2's lifecycle manager reports every node ACTIVE (the action servers exist
        before that: the first plan request, 1 s after the costmaps started, failed). Nav2 only
        finishes activating once the simulator publishes TF, so `tick` keeps it publishing."""
        t0, fut, last = time.time(), None, 0.0
        while time.time() - t0 < timeout:
            if tick is not None:
                tick()
            rclpy.spin_once(self.node, timeout_sec=0.01)
            if fut is not None and fut.done():
                if fut.result() is not None and fut.result().success and self.nav_client.server_is_ready() \
                        and self.path_client.server_is_ready():
                    return True
                fut = None
            if fut is None and time.time() - last > 0.5 and self.active_srv.service_is_ready():
                fut, last = self.active_srv.call_async(Trigger.Request()), time.time()
        return False

    def plan_length(self, start, goal, tick=None, timeout=10.0):
        """Length (m) of the path Nav2's planner finds from start (x, y, yaw) to goal, inf if none."""
        req = ComputePathToPose.Goal()
        req.start, req.goal, req.use_start = self._pose(*start), self._pose(*goal), True
        fut = self.path_client.send_goal_async(req)
        res = self._wait(fut, tick, timeout)
        if res is None or not res.accepted:
            return math.inf, None
        rfut = res.get_result_async()
        out = self._wait(rfut, tick, timeout)
        if out is None or not out.result.path.poses:
            return math.inf, None
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in out.result.path.poses])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), pts

    def navigate(self, goal, tick=None):
        """Send NavigateToPose (x, y, yaw); returns immediately. Poll done() while stepping."""
        self.result = None
        req = NavigateToPose.Goal()
        req.pose = self._pose(*goal)
        fut = self.nav_client.send_goal_async(req)
        gh = self._wait(fut, tick, 10.0)
        self.goal_handle = gh if gh is not None and gh.accepted else None
        if self.goal_handle is None:
            return False
        self._result_fut = self.goal_handle.get_result_async()
        return True

    def navigate_through(self, poses, tick=None):
        """Send NavigateThroughPoses [(x, y, yaw), ...]: one path through all of them, stopping only at
        the last (the goal checker applies there). Poll done() while stepping."""
        self.result = None
        req = NavigateThroughPoses.Goal()
        req.poses = [self._pose(*p) for p in poses]
        fut = self.through_client.send_goal_async(req)
        gh = self._wait(fut, tick, 10.0)
        self.goal_handle = gh if gh is not None and gh.accepted else None
        if self.goal_handle is None:
            return False
        self._result_fut = self.goal_handle.get_result_async()
        return True

    def dock(self, dock_pose, dock_type="pick_point", tick=None):
        """DockRobot at dock_pose (x, y, yaw), robot already at the staging pose (navigate_to_staging_pose
        False): the docking server's controller drives the last metres. Poll done(); result() has the
        error code."""
        self.result = None
        req = DockRobot.Goal()
        req.use_dock_id, req.dock_type, req.navigate_to_staging_pose = False, dock_type, False
        req.dock_pose = self._pose(*dock_pose)
        fut = self.dock_client.send_goal_async(req)
        gh = self._wait(fut, tick, 10.0)
        self.goal_handle = gh if gh is not None and gh.accepted else None
        if self.goal_handle is None:
            return False
        self._result_fut = self.goal_handle.get_result_async()
        return True

    # ---------------------------------------------------------------- streamed goals (the DualVLN walk)
    def _path_msg(self, pts):
        """[(x, y), ...] in the map -> nav_msgs/Path, each pose facing the next point."""
        m = Path()
        m.header.frame_id, m.header.stamp = "map", _stamp(self.t)
        for i, (x, y) in enumerate(pts):
            j = min(i + 1, len(pts) - 1)
            yaw = math.atan2(pts[j][1] - pts[i][1], pts[j][0] - pts[i][0]) if j != i else \
                math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0])
            m.poses.append(self._pose(x, y, yaw))
        return m

    def stream_path(self, pts, controller="FollowPath"):
        """FollowPath on the controller server, without waiting: Nav2's controller follows THIS path with its
        collision checking; a new call replaces the old path (the server preempts)."""
        req = FollowPath.Goal()
        req.path, req.controller_id = self._path_msg(pts), controller
        self._stream = self._stream[-3:] + [self.follow_client.send_goal_async(req)]

    def stream_goal(self, pose):
        """NavigateToPose without waiting: Nav2 plans its own path there; a new call replaces the goal."""
        req = NavigateToPose.Goal()
        req.pose = self._pose(*pose)
        self._stream = self._stream[-3:] + [self.nav_client.send_goal_async(req)]

    def stop_stream(self):
        """Cancel whatever streamed goal is still running."""
        for fut in self._stream:
            if fut.done() and fut.result() is not None and fut.result().accepted:
                fut.result().cancel_goal_async()
        self._stream = []

    def last_result(self):
        """The finished action's result message (DockRobot: success, error_code, error_msg), or None."""
        if self.goal_handle is None or not self._result_fut.done():
            return None
        return self._result_fut.result().result

    def done(self):
        """None while navigating, else the Nav2 status code (4 = succeeded, 5 = canceled, 6 = aborted)."""
        if self.goal_handle is None:
            return 6
        if self._result_fut.done():
            return self._result_fut.result().status
        return None

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    def _wait(self, fut, tick, timeout):
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            if tick is not None:
                tick()
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return fut.result() if fut.done() else None

    def close(self):
        """Destroy the node and shut ROS down: left open, it kept the Isaac process from exiting."""
        try:
            self.node.destroy_node()
        finally:
            rclpy.try_shutdown()
