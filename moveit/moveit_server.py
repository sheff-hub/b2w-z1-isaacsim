#!/usr/bin/env python3
"""HTTP bridge between the simulator and MoveIt's move_group (runs in the b2w-moveit container next to it).

Isaac Sim's bundled ROS 2 libraries have no MoveIt messages, so the simulator talks to this server in JSON (like
the DualVLN server) and the server talks to move_group in ROS 2. MoveIt only PLANS; the simulator executes the
returned joint trajectories itself (project/moveit_link.py).

Everything is in the arm's base frame, link00. Poses are [x, y, z, qw, qx, qy, qz]; joint vectors are joint1..6.

    GET  /health
    POST /scene      {"objects": [{"id", "size": [sx, sy, sz], "pose": [...]}, ...],
                      "attached": [{"id", "link", "size", "pose" (in that link's frame), "touch_links": [...]}]}
                     replaces whatever the previous /scene call set up
    POST /plan       {"start": [6], "gripper": rad, "goal": {"pose": [...]} | {"joints": [6]},
                      "pos_tol": m, "rot_tol": rad, "time": s, "attempts": n, "vel": 0..1, "acc": 0..1}
    POST /cartesian  {"start": [6], "gripper": rad, "waypoints": [[...], ...], "max_step": m,
                      "avoid_collisions": bool, "vel": 0..1, "acc": 0..1}
    POST /approach   {"start": [6], "pre": [...], "final": [...], "gripper": rad, "approach_gripper": rad, "tries": n}
                     -> {"joints": [6]}: an IK solution for the pre-pose from which the straight line to the final
                     pose works (collision-checked), the nearest to the start of those found -- plan there with
                     /plan {"goal": {"joints"}}, then /cartesian. A pose goal alone may end in a posture (elbow
                     folded on its stop) from which the straight approach is impossible.
    POST /check      {"state": [6], "gripper": rad} -> {"valid", "contacts": [[a, b], ...]}
    POST /fk         {"state": [6], "link": name} -> {"pose": [...]}  (to check MoveIt's model against the simulator)
Replies with a trajectory: {"ok", "code", "names", "points": [[t, q1..q6], ...], ...}.
"""
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, BoundingVolume, CollisionObject, Constraints, JointConstraint,
                             MotionPlanRequest, OrientationConstraint, PlanningOptions, PlanningScene,
                             PositionConstraint, RobotState, WorkspaceParameters)
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionFK, GetPositionIK, GetStateValidity
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

FRAME, GROUP, TIP = "link00", "arm", "grasp_point"
ARM = [f"joint{i}" for i in range(1, 7)]
# random IK seeds inside the simulator's joint limits (moveit/make_config.py: SIM_LIMITS_DEG), a little in from them
SEED_RANGES = [(-2.5, 2.5), (0.1, 2.8), (-2.7, -0.1), (-1.4, 1.4), (-1.2, 1.2), (-2.6, 2.6)]
PORT = int(os.environ.get("MOVEIT_PORT", "5810"))


def pose_msg(p):
    m = Pose()
    m.position.x, m.position.y, m.position.z = (float(v) for v in p[:3])
    m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = (float(v) for v in p[3:7])
    return m


def robot_state(q, gripper):
    rs = RobotState()
    rs.joint_state = JointState(name=ARM + ["jointGripper"], position=[float(v) for v in q] + [float(gripper)])
    rs.is_diff = True  # keep the scene's attached objects (False would detach them for this request)
    return rs


def box_object(oid, size, pose, frame=FRAME, op=CollisionObject.ADD):
    co = CollisionObject()
    co.header.frame_id = frame
    co.id = oid
    co.operation = op
    if op == CollisionObject.ADD:
        co.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in size])]
        co.primitive_poses = [pose_msg(pose)]
    return co


def trajectory_reply(traj):
    jt = traj.joint_trajectory
    idx = [jt.joint_names.index(n) for n in ARM]
    pts = [[p.time_from_start.sec + p.time_from_start.nanosec * 1e-9] + [p.positions[i] for i in idx]
           for p in jt.points]
    return {"names": ARM, "points": pts}


class Bridge:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("z1_moveit_bridge")
        self.move = ActionClient(self.node, MoveGroup, "/move_action")
        self.cart = self.node.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.apply = self.node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.valid = self.node.create_client(GetStateValidity, "/check_state_validity")
        self.fk_srv = self.node.create_client(GetPositionFK, "/compute_fk")
        self.ik_srv = self.node.create_client(GetPositionIK, "/compute_ik")
        # move_group's current-state monitor waits up to 1 s per request for /joint_states; publish the last state
        self.js_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.last_js = [0.0] * 7
        self.node.create_timer(0.05, self._publish_js)
        self.ex = MultiThreadedExecutor()
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()
        self.world_ids, self.attached_ids = set(), set()
        self.lock = threading.Lock()  # one request at a time: the planning scene is shared state

    def _publish_js(self):
        msg = JointState(name=ARM + ["jointGripper"], position=list(self.last_js))
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.js_pub.publish(msg)

    def wait(self, future, timeout=60.0):
        ev = threading.Event()
        future.add_done_callback(lambda _: ev.set())
        if not ev.wait(timeout):
            raise TimeoutError("MoveIt did not answer")
        return future.result()

    def ready(self):
        return (self.move.wait_for_server(timeout_sec=0.5) and self.cart.wait_for_service(timeout_sec=0.5)
                and self.apply.wait_for_service(timeout_sec=0.5))

    def _apply(self, ps):
        ps.is_diff = True
        ps.robot_state.is_diff = True
        r = self.wait(self.apply.call_async(ApplyPlanningScene.Request(scene=ps)))
        if not r.success:
            raise RuntimeError("apply_planning_scene failed")

    def scene(self, req):
        with self.lock:
            # 1) detach and drop what the previous call attached, 2) drop the previous world objects
            if self.attached_ids:
                ps = PlanningScene()
                for oid in self.attached_ids:
                    aco = AttachedCollisionObject(object=box_object(oid, None, None, op=CollisionObject.REMOVE))
                    ps.robot_state.attached_collision_objects.append(aco)
                self._apply(ps)
                ps = PlanningScene()
                ps.world.collision_objects = [box_object(oid, None, None, op=CollisionObject.REMOVE)
                                              for oid in self.attached_ids]
                self._apply(ps)
            if self.world_ids:
                ps = PlanningScene()
                ps.world.collision_objects = [box_object(oid, None, None, op=CollisionObject.REMOVE)
                                              for oid in self.world_ids]
                self._apply(ps)
            # 3) the new world and attached objects
            ps = PlanningScene()
            ps.world.collision_objects = [box_object(o["id"], o["size"], o["pose"]) for o in req.get("objects", [])]
            self._apply(ps)
            self.world_ids = {o["id"] for o in req.get("objects", [])}
            ps = PlanningScene()
            for a in req.get("attached", []):
                aco = AttachedCollisionObject(link_name=a["link"],
                                              object=box_object(a["id"], a["size"], a["pose"], frame=a["link"]),
                                              touch_links=a.get("touch_links", []))
                ps.robot_state.attached_collision_objects.append(aco)
            if ps.robot_state.attached_collision_objects:
                self._apply(ps)
            self.attached_ids = {a["id"] for a in req.get("attached", [])}
            return {"ok": True, "objects": len(self.world_ids), "attached": len(self.attached_ids)}

    def plan(self, req):
        mpr = MotionPlanRequest()
        mpr.workspace_parameters = WorkspaceParameters()
        mpr.workspace_parameters.header.frame_id = FRAME
        mpr.workspace_parameters.min_corner.x = mpr.workspace_parameters.min_corner.y = -2.0
        mpr.workspace_parameters.min_corner.z = -2.0
        mpr.workspace_parameters.max_corner.x = mpr.workspace_parameters.max_corner.y = 2.0
        mpr.workspace_parameters.max_corner.z = 2.0
        mpr.start_state = robot_state(req["start"], req.get("gripper", 0.0))
        mpr.group_name = GROUP
        mpr.pipeline_id = "ompl"
        mpr.planner_id = req.get("planner", "RRTConnect")
        mpr.num_planning_attempts = int(req.get("attempts", 5))
        mpr.allowed_planning_time = float(req.get("time", 5.0))
        mpr.max_velocity_scaling_factor = float(req.get("vel", 0.3))
        mpr.max_acceleration_scaling_factor = float(req.get("acc", 0.3))
        c = Constraints()
        goal = req["goal"]
        if "joints" in goal:
            for n, v in zip(ARM, goal["joints"]):
                c.joint_constraints.append(JointConstraint(joint_name=n, position=float(v), tolerance_above=0.002,
                                                           tolerance_below=0.002, weight=1.0))
        else:
            p = goal["pose"]
            pc = PositionConstraint()
            pc.header.frame_id = FRAME
            pc.link_name = goal.get("link", TIP)
            pc.constraint_region = BoundingVolume(
                primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[float(req.get("pos_tol", 0.002))])],
                primitive_poses=[pose_msg(p)])
            pc.weight = 1.0
            oc = OrientationConstraint()
            oc.header.frame_id = FRAME
            oc.link_name = goal.get("link", TIP)
            oc.orientation = pose_msg(p).orientation
            tol = float(req.get("rot_tol", 0.02))
            oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = tol
            oc.weight = 1.0
            c.position_constraints, c.orientation_constraints = [pc], [oc]
        mpr.goal_constraints = [c]
        opts = PlanningOptions(plan_only=True)
        opts.planning_scene_diff.is_diff = True
        opts.planning_scene_diff.robot_state.is_diff = True
        self.last_js = [float(v) for v in req["start"]] + [float(req.get("gripper", 0.0))]
        tries = 0
        with self.lock:
            # OMPL is randomised: a path whose re-timed points fail the final collision check (ValidateSolution,
            # code FAILURE) usually has a valid neighbour, so plan again (req["retries"] times)
            for tries in range(1, int(req.get("retries", 3)) + 1):
                gh = self.wait(self.move.send_goal_async(MoveGroup.Goal(request=mpr, planning_options=opts)))
                if not gh.accepted:
                    return {"ok": False, "code": -1, "error": "goal rejected"}
                res = self.wait(gh.get_result_async(), timeout=float(req.get("time", 5.0)) * 4 + 30).result
                if res.error_code.val == 1:
                    break
        out = {"ok": res.error_code.val == 1, "code": res.error_code.val, "plan_time": round(res.planning_time, 3),
               "tries": tries}
        if out["ok"]:
            out.update(trajectory_reply(res.planned_trajectory))
        return out

    def cartesian(self, req):
        r = GetCartesianPath.Request()
        r.header.frame_id = FRAME
        r.start_state = robot_state(req["start"], req.get("gripper", 0.0))
        r.group_name, r.link_name = GROUP, req.get("link", TIP)
        r.waypoints = [pose_msg(w) for w in req["waypoints"]]
        r.max_step = float(req.get("max_step", 0.005))
        r.jump_threshold = 0.0
        r.prismatic_jump_threshold = 0.0
        r.revolute_jump_threshold = float(req.get("revolute_jump", 0.0))
        r.avoid_collisions = bool(req.get("avoid_collisions", True))
        r.max_velocity_scaling_factor = float(req.get("vel", 0.2))
        r.max_acceleration_scaling_factor = float(req.get("acc", 0.2))
        self.last_js = [float(v) for v in req["start"]] + [float(req.get("gripper", 0.0))]
        with self.lock:
            res = self.wait(self.cart.call_async(r))
        out = {"ok": res.error_code.val == 1 and res.fraction > 0.999, "code": res.error_code.val,
               "fraction": round(res.fraction, 4)}
        if res.solution.joint_trajectory.points:
            out.update(trajectory_reply(res.solution))
        return out

    def approach(self, req):
        grip, agrip = float(req.get("gripper", 0.0)), float(req.get("approach_gripper", req.get("gripper", 0.0)))
        start = [float(v) for v in req["start"]]
        found, tried = [], 0
        for k in range(int(req.get("tries", 12))):
            tried += 1
            seed = start if k == 0 else [random.uniform(a, b) for a, b in SEED_RANGES]
            ikr = GetPositionIK.Request()
            ikr.ik_request.group_name = GROUP
            ikr.ik_request.robot_state = robot_state(seed, grip)
            ikr.ik_request.avoid_collisions = True
            ikr.ik_request.ik_link_name = TIP
            ikr.ik_request.pose_stamped.header.frame_id = FRAME
            ikr.ik_request.pose_stamped.pose = pose_msg(req["pre"])
            ikr.ik_request.timeout = Duration(sec=0, nanosec=100_000_000)
            with self.lock:
                res = self.wait(self.ik_srv.call_async(ikr))
            if res.error_code.val != 1:
                continue
            js = res.solution.joint_state
            q = [js.position[js.name.index(n)] for n in ARM]
            cr = self.cartesian({"start": q, "gripper": agrip, "waypoints": [req["final"]],
                                 "max_step": float(req.get("max_step", 0.005)),
                                 "avoid_collisions": bool(req.get("avoid_collisions", True))})
            if cr["ok"]:
                found.append((sum((a - b) ** 2 for a, b in zip(q, start)), q))
                if len(found) >= int(req.get("enough", 3)):
                    break
        if not found:
            return {"ok": False, "code": -31, "tried": tried, "error": "no pre-pose with a straight approach"}
        q = min(found)[1]
        return {"ok": True, "code": 1, "joints": q, "tried": tried, "found": len(found)}

    def check(self, req):
        r = GetStateValidity.Request(robot_state=robot_state(req["state"], req.get("gripper", 0.0)), group_name=GROUP)
        with self.lock:
            res = self.wait(self.valid.call_async(r))
        pairs = sorted({tuple(sorted((c.contact_body_1, c.contact_body_2))) for c in res.contacts})
        return {"valid": res.valid, "contacts": [list(p) for p in pairs]}


    def fk(self, req):
        r = GetPositionFK.Request(fk_link_names=[req.get("link", TIP)],
                                  robot_state=robot_state(req["state"], req.get("gripper", 0.0)))
        r.header.frame_id = FRAME
        with self.lock:
            res = self.wait(self.fk_srv.call_async(r))
        if not res.pose_stamped:
            return {"ok": False, "code": res.error_code.val}
        p = res.pose_stamped[0].pose
        return {"ok": True, "pose": [p.position.x, p.position.y, p.position.z,
                                     p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]}


BRIDGE = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ok = BRIDGE.ready()
            self._send(200 if ok else 503, {"ok": ok})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        fn = {"/scene": BRIDGE.scene, "/plan": BRIDGE.plan, "/cartesian": BRIDGE.cartesian,
              "/check": BRIDGE.check, "/fk": BRIDGE.fk, "/approach": BRIDGE.approach}.get(self.path)
        if fn is None:
            return self._send(404, {"error": "unknown path"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            t0 = time.time()
            out = fn(req)
            out["server_s"] = round(time.time() - t0, 3)
            self._send(200, out)
        except Exception as e:  # noqa: BLE001 -- reported to the caller, which decides what to do
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # one short line per request
        print(f"[moveit_server] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


def main():
    global BRIDGE
    BRIDGE = Bridge()
    for _ in range(120):
        if BRIDGE.ready():
            break
        time.sleep(1.0)
    print(f"[moveit_server] move_group {'ready' if BRIDGE.ready() else 'NOT ready'}; listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
