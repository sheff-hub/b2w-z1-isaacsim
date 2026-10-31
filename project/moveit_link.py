"""MoveIt client for the simulator side (fetch_box.py --arm moveit).

MoveIt runs in its own container (moveit/: move_group + the HTTP bridge moveit_server.py); Isaac Sim's bundled
ROS 2 has no MoveIt messages, so this talks JSON over HTTP. MoveIt PLANS in the arm's base frame (link00); the
simulator EXECUTES the returned joint trajectories (fetch_box.run_traj). Poses are [x, y, z, qw, qx, qy, qz].
"""
import requests

# The robot's own body as boxes in the B2W base frame (x forward, z up), for MoveIt to keep the arm out of:
# (name, centre, size). Body box, top cover and lidar cage are the measured colliders (see moveit/README.md);
# hips, head cameras and thighs are approximated around the link positions of the standing robot.
BODY_BOXES = [
    ("body", (0.0, 0.0, 0.0), (0.50, 0.28, 0.15)),
    ("top_cover", (0.0, 0.0, 0.070), (0.80, 0.28, 0.03)),
    ("front_hips_head", (0.355, 0.0, 0.0), (0.21, 0.30, 0.16)),   # FL/FR hip motors, D455 head camera
    ("rear_hips", (-0.355, 0.0, 0.0), (0.21, 0.30, 0.16)),
    ("lidar_cage", (0.345, 0.0, 0.170), (0.16, 0.16, 0.34)),       # Ouster OS0 on its cage, in front of the arm
    ("thigh_fl", (0.33, 0.20, -0.08), (0.14, 0.08, 0.28)),
    ("thigh_fr", (0.33, -0.20, -0.08), (0.14, 0.08, 0.28)),
    ("thigh_rl", (-0.33, 0.20, -0.08), (0.14, 0.08, 0.28)),
    ("thigh_rr", (-0.33, -0.20, -0.08), (0.14, 0.08, 0.28)),
]
TOUCH_LINKS = ["grasp_point", "gripperStator", "gripperMover", "link06"]  # may touch a box held in the gripper


class MoveIt:
    def __init__(self, url="http://127.0.0.1:5810", timeout=120.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _post(self, path, body):
        r = requests.post(self.url + path, json=body, timeout=self.timeout)
        out = r.json()
        if r.status_code != 200 and "error" not in out:
            out["error"] = f"HTTP {r.status_code}"
        return out

    def health(self):
        try:
            return bool(requests.get(self.url + "/health", timeout=3).json().get("ok"))
        except requests.RequestException:
            return False

    def scene(self, objects, attached=()):
        """objects: [(id, centre-pose [7], size [3])] in link00; attached: [(id, link, pose in link frame, size)]."""
        return self._post("/scene", {
            "objects": [{"id": i, "pose": list(p), "size": list(s)} for i, p, s in objects],
            "attached": [{"id": i, "link": ln, "pose": list(p), "size": list(s), "touch_links": TOUCH_LINKS}
                         for i, ln, p, s in attached]})

    def plan_pose(self, start, pose, gripper=0.0, **kw):
        return self._post("/plan", dict(kw, start=list(start), gripper=gripper, goal={"pose": list(pose)}))

    def plan_joints(self, start, joints, gripper=0.0, **kw):
        return self._post("/plan", dict(kw, start=list(start), gripper=gripper, goal={"joints": list(joints)}))

    def cartesian(self, start, waypoints, gripper=0.0, **kw):
        return self._post("/cartesian", dict(kw, start=list(start), gripper=gripper,
                                             waypoints=[list(w) for w in waypoints]))

    def approach(self, start, pre, final, gripper=0.0, approach_gripper=None, **kw):
        """Joints for the pre-pose from which the straight line to `final` works (see moveit_server.py)."""
        return self._post("/approach", dict(kw, start=list(start), pre=list(pre), final=list(final), gripper=gripper,
                                             approach_gripper=gripper if approach_gripper is None else approach_gripper))

    def check(self, state, gripper=0.0):
        """Is this arm state collision-free in the current planning scene? {"valid", "contacts": [[a, b], ...]}"""
        return self._post("/check", {"state": list(state), "gripper": gripper})

    def fk(self, state, gripper=0.0, link="grasp_point"):
        return self._post("/fk", {"state": list(state), "gripper": gripper, "link": link})
