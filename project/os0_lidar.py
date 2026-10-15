"""The robot's own 3D lidar (the RTX "OS0" sensor in b2w_z1_gripper.usd, on lidar_link: 128 channels,
360 deg, +-44.5 deg vertical, 0.3-50 m) as a point cloud for Nav2.

Import AFTER the AppLauncher has started; the app must render (--enable_cameras), as for cameras,
and isaacsim.sensors.rtx must be enabled AT START-UP (--kit_args "--enable isaacsim.sensors.rtx"):
it registers the lidar annotators.
Reads the sensor through omni.replicator's IsaacExtractRTXSensorPointCloudNoAccumulator annotator
(the points fired since the last rendered frame), accumulates them for one publishing period,
transforms them into the world (= odom) frame, drops the robot's own body (a box in the base
frame) and returns them; nav2_link publishes them as sensor_msgs/PointCloud2.
"""

import math

import numpy as np

import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom

ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"
# the robot itself, in the base frame (m): body + wheels + stowed arm + lidar mast. Points inside are
# self-hits; on a real robot this is a laser/point-cloud box filter.
SELF_BOX = ((-0.70, 0.75), (-0.45, 0.45), (-0.80, 0.70))


def find_lidar(root="/World/envs/env_0/Robot"):
    stage = omni.usd.get_context().get_stage()
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if str(p.GetTypeName()) == "OmniLidar":
            return str(p.GetPath())
    return None


def quat_to_mat(q):
    """(w, x, y, z) -> 3x3."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class OS0:
    """link_pose_fn() -> (position (3,), quaternion w,x,y,z (4,)) of the body carrying the sensor
    (lidar_link), in the world, FROM PHYSICS. The USD transforms of the robot's links are not updated
    while Isaac Lab simulates (only PhysX / Fabric are), so the USD pose of the sensor stays at the
    spawn: used for the transforms, it put every cloud back at the start position (Nav2: "Sensor origin
    at (1.41, -1.50) is out of map bounds", false obstacles round the spawn, no plans)."""

    def __init__(self, link_pose_fn, prim_path=None, link_path=None):
        self.path = prim_path or find_lidar()
        if self.path is None:
            raise RuntimeError("no OmniLidar prim under the robot")
        # Isaac Lab renders during env.step only when it knows of an RTX sensor -- it sets this flag for
        # sensors made by its own classes, not for one that comes with the robot's USD
        import carb
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
        self.rp = rep.create.render_product(self.path, [1, 1], name="os0_lidar")
        self.annot = rep.AnnotatorRegistry.get_annotator(ANNOTATOR)
        self.annot.attach([self.rp])
        self.buf = []
        self.frame_note = None
        self.link_pose_fn = link_pose_fn
        # the sensor's fixed offset from its link, from the USD (both poses are the spawn ones, so the
        # RELATIVE transform is right): world = sensor_in_link (x) link_world
        stage = omni.usd.get_context().get_stage()
        link_path = link_path or self.path.split("/lidar_link")[0] + "/lidar_link"
        Ms = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(self.path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        Ml = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(link_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        self.M_in_link = Ms @ np.linalg.inv(Ml)  # row-vector convention

    def sensor_world(self):
        """4x4 world transform of the sensor NOW (row-vector convention, as USD): link pose from physics."""
        pos, quat = self.link_pose_fn()
        Ml = np.eye(4)
        Ml[:3, :3] = quat_to_mat(np.asarray(quat, dtype=np.float64)).T  # row-vector: rows are the link's axes
        Ml[3, :3] = np.asarray(pos, dtype=np.float64)
        return self.M_in_link @ Ml

    def grab(self):
        """Call after each rendered step: collect the points fired since the last frame (sensor frame)."""
        d = self.annot.get_data()
        pts = d.get("data") if isinstance(d, dict) else d
        if pts is None or len(pts) == 0:
            return 0
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        self.buf.append(pts)
        if self.frame_note is None:
            info = d.get("info", {}) if isinstance(d, dict) else {}
            self.frame_note = {"n": len(pts), "info_keys": list(info.keys())[:8]}
        return len(pts)

    def take(self, voxel=0.08, zmin_world=-0.3, zmax_world=2.0, rmax=6.0):
        """All points since the last take(), in the SENSOR frame, voxel-downsampled and limited to a world
        height band (floor-level noise and the ceiling dropped), plus the sensor's world matrix (USD
        row-vector 4x4). The annotator reports sensor-frame points: p_world = p @ M[:3,:3] + M[3,:3]
        (checked against the collider map, os0_probe.py)."""
        T = self.sensor_world()
        if not self.buf:
            return np.zeros((0, 3), np.float32), T
        pts = np.concatenate(self.buf)
        self.buf = []
        zw = pts @ T[:3, 2] + T[3, 2]
        keep = (zw > zmin_world) & (zw < zmax_world)
        if rmax:  # Nav2 uses obstacles to 5 m: farther points only cost CPU in the costmaps
            keep &= (pts[:, 0] ** 2 + pts[:, 1] ** 2) < rmax * rmax
        pts = pts[keep]
        if voxel and len(pts):
            key = np.floor(pts / voxel).astype(np.int32)
            _, idx = np.unique(key, axis=0, return_index=True)
            pts = pts[idx]
        return pts.astype(np.float32), T
