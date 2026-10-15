"""An Intel RealSense D405 on a ring clamp around the Z1's link06 (section I of
build_b2w_z1_gripper_asset.py), plus offline checks that need no simulator or GPU.

    repos/3dgrut/.venv/bin/python tools/wrist_d405.py [--usd robots_usd/b2w_z1_gripper.usd]
        -> clearance and field-of-view report

THE DESIGN (link06 frame: x along the wrist, z the side the moving jaw opens to), after the user's sketch
(13 Oct 2026) and the 3D-printed Z1 end mounts people share (e.g. MakerWorld 430201, a D455 + AprilTag
mount for the Z1's end):
- Camera: RealSense D405, Intel's own mesh (robots_usd/RealSense/d405.usd, made by make_d405_usd.py from
  realsense-ros' d405.stl): 42 x 42 x 23 mm, 60 g, stereo pair 18 mm apart whose LEFT imager also gives
  the colour image; 87 x 58 deg, depth from 7 cm. The USD camera sits on the left imager, which is also
  the asset's origin.
- Camera pose: where the user put it by hand in robots_usd/b2w_z1_gripper_reference.usd -- 17.6 cm out on
  the gripper's -y side, a little ahead of link06, turned 27.7 deg toward the gripper and 5.5 deg down.
  From there the whole gripper is in view: jaw pivot 15 cm ahead (38 deg off axis), fingertips 22 cm
  ahead (20 deg). The -y side is also the one that faces up and away at a grasp (+y faces the box).
- Holder, 3D-printed (PETG, black), all clamped to link06 so it rolls with the wrist:
    ring     a split collar round link06 (a 32.5 mm cylinder from x 0 to 0.045), on its outer part
             (x 26-43 mm): with the arm stowed, link02 tucks in under link06's first 25 mm (1 mm away), so
             a ring in the middle pressed on it (17.6 N, probe_self_collision). Bore 33 mm, 7.5 mm wall,
             17 mm wide, two ears and an M3 bolt at the split (+z);
    arm      a 9 x 18 mm beam from the ring's -y side to the camera, angled to meet it;
    cradle   a box round the camera: back, top, bottom and both side walls 2.5 mm, 0.5 mm clearance, a 2 mm
             lip round the front face (clear of the lenses), and a pad on its inner side for the arm.
- Nothing reaches the moving jaw (x >= 0.079); the ring ends 1 mm before the fixed jaw's housing (x 0.044),
  which is bolted to link06 anyway.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

Z1 = "/Unitree_B2W/z1"
STATOR = f"{Z1}/gripperStator"
MOVER = f"{Z1}/gripperMover"
LINK06 = f"{Z1}/link06"
EE_CAMERA = f"{LINK06}/end_effector_camera"
HOLDER = f"{LINK06}/wrist_d405"
D405_ASSET = "./RealSense/d405.usd"  # relative to robots_usd/ (the robot file's folder)
STATOR_X = 0.051  # gripperStator frame = link06 frame shifted this far along x (the recorded EE frame)
PIVOT = np.array([0.100, 0.0, 0.0])  # jointGripper axis (y) in the link06 frame
GRIPPER_OPEN = -1.4  # rad, fetch_env_cfg.GRIPPER_OPEN

# D405, camera frame (origin = left imager, x forward, y left, z up): body extent of Intel's mesh
BODY_MIN = np.array([-0.0192, -0.0301, -0.0210])
BODY_MAX = np.array([0.0038, 0.0120, 0.0210])
CAM_MASS = 0.060  # kg
HFOV, VFOV = 87.0, 58.0  # deg; at 16:9 the vertical comes out 56 deg
APERTURE_H = 20.955  # mm (any; the focal length sets the field of view)
FOCAL = APERTURE_H / 2 / math.tan(math.radians(HFOV / 2))  # 11.04 mm
APERTURE_V = APERTURE_H * 9 / 16  # 16:9, the D405's native aspect

# Camera pose (link06): the user's placement in b2w_z1_gripper_reference.usd
EYE = np.array([0.0206, -0.1760, 0.0142])
YAW, PITCH = math.radians(27.7), math.radians(-5.5)  # toward the gripper axis (+y), down

# Ring clamp round link06
LINK06_R = 0.0325
RING_X = (0.026, 0.043)  # outer part of link06 (cylinder x 0 .. 0.045): stowed, link02 tucks under x < 0.025
RING_RI, RING_RO = LINK06_R + 0.0005, LINK06_R + 0.0080
SPLIT_GAP = 0.002  # m, at +z
EAR = (0.012, 0.005, 0.012)  # ear size: along x, thick (y), out from the ring (z)
BOLT_R = 0.0015
# Arm and cradle
ARM_W, ARM_H = 0.009, 0.018  # beam width (in the x-y plane) and height (z)
ARM_Z = 0.003  # beam centre height
WALL, CLEAR, LIP, LIP_T = 0.0025, 0.0005, 0.002, 0.0015  # cradle wall, clearance, front lip width / thickness
PAD_T = 0.005
PETG = 0.6 * 1270.0  # kg/m^3: printed PETG, solid walls + 25 % infill is ~60 % solid
BLACK, METAL = (0.05, 0.05, 0.06), (0.62, 0.64, 0.66)


def camera_frame():
    """Left imager position and rotation (cols: forward, left, up) in the link06 frame."""
    f = np.array([math.cos(PITCH) * math.cos(YAW), math.cos(PITCH) * math.sin(YAW), math.sin(PITCH)])
    up = np.array([-math.sin(PITCH) * math.cos(YAW), -math.sin(PITCH) * math.sin(YAW), math.cos(PITCH)])
    left = np.cross(up, f)
    return EYE.copy(), np.stack([f, left, up], axis=1)


def design():
    """Collider/visual boxes in the link06 frame and the camera pose.
    Box = (name, centre, R (cols = local x, y, z), size, collider, colour or None (collider only), density or None)."""
    o, R = camera_frame()
    lo, hi = BODY_MIN - CLEAR, BODY_MAX + CLEAR  # cradle cavity (camera frame)
    boxes = []

    def cbox(name, a, b, col=True, rgb=BLACK, rho=PETG):  # a box given in the camera frame
        a, b = np.asarray(a, float), np.asarray(b, float)
        boxes.append((name, o + R @ ((a + b) / 2), R, tuple(b - a), col, rgb, rho))

    W = WALL
    cbox("cradle_back", [lo[0] - W, lo[1] - W, lo[2] - W], [lo[0], hi[1] + W, hi[2] + W])
    cbox("cradle_bottom", [lo[0], lo[1] - W, lo[2] - W], [hi[0], hi[1] + W, lo[2]])
    cbox("cradle_top", [lo[0], lo[1] - W, hi[2]], [hi[0], hi[1] + W, hi[2] + W])
    cbox("cradle_outer", [lo[0], lo[1] - W, lo[2]], [hi[0], lo[1], hi[2]])
    cbox("cradle_inner", [lo[0], hi[1], lo[2]], [hi[0], hi[1] + W, hi[2]])
    f0, f1 = hi[0], hi[0] + LIP_T  # the lip, in front of the front face
    cbox("lip_top", [f0, lo[1] - W, hi[2] - LIP], [f1, hi[1] + W, hi[2] + W])
    cbox("lip_bottom", [f0, lo[1] - W, lo[2] - W], [f1, hi[1] + W, lo[2] + LIP])
    cbox("lip_outer", [f0, lo[1] - W, lo[2] + LIP], [f1, lo[1] + LIP, hi[2] - LIP])
    cbox("lip_inner", [f0, hi[1] - LIP, lo[2] + LIP], [f1, hi[1] + W, hi[2] - LIP])
    pad_a = np.array([lo[0] - W, hi[1] + W, -0.6 * (hi[2] + W)])
    pad_b = np.array([hi[0] - 0.006, hi[1] + W + PAD_T, 0.6 * (hi[2] + W)])
    cbox("arm_pad", pad_a, pad_b)
    cbox("camera_collider", BODY_MIN, BODY_MAX, rgb=None, rho=None)  # the D405 itself (mass CAM_MASS)

    # the arm: from inside the ring wall to the middle of the pad, in the x-y plane
    pad_c = o + R @ ((pad_a + pad_b) / 2)
    xm = np.mean(RING_X)
    end = np.array([pad_c[0], pad_c[1], ARM_Z])
    dxy = end[:2] - np.array([xm, 0.0])
    dxy /= np.linalg.norm(dxy)
    rm = (RING_RI + RING_RO) / 2
    start = np.array([xm + dxy[0] * rm, dxy[1] * rm, ARM_Z])
    ay = np.array([*(end - start)[:2], 0.0])
    length = np.linalg.norm(ay)
    ay /= length
    az = np.array([0.0, 0.0, 1.0])
    Ra = np.stack([np.cross(ay, az), ay, az], axis=1)
    boxes.append(("arm", (start + end) / 2, Ra, (ARM_W, length, ARM_H), True, BLACK, PETG))

    # clamp ears either side of the split at +z (the bolt runs through them along y)
    I = np.eye(3)
    for s, name in ((1, "ear_left"), (-1, "ear_right")):
        c = np.array([xm, s * (SPLIT_GAP / 2 + EAR[1] / 2), RING_RO - 0.002 + EAR[2] / 2])
        boxes.append((name, c, I, EAR, True, BLACK, PETG))
    # the ring: collider boxes round it (a ring is not convex); its visual is a mesh, its mass ring_mass()
    n = 16
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        c = np.array([xm, math.cos(a) * rm, math.sin(a) * rm])
        Rk = np.stack([np.array([1.0, 0, 0]), np.array([0, math.cos(a), math.sin(a)]),
                       np.array([0, -math.sin(a), math.cos(a)])], axis=1)  # cols: x, radial, tangent (det +1)
        chord = 2 * RING_RO * math.tan(math.pi / n)
        boxes.append((f"ring_collider_{k:02d}", c, Rk, (RING_X[1] - RING_X[0], RING_RO - RING_RI, chord), True,
                      None, None))
    return {"boxes": boxes, "R": R, "eye": o, "forward": R[:, 0]}


def ring_mass():
    gap_angle = SPLIT_GAP / ((RING_RI + RING_RO) / 2)
    return math.pi * (RING_RO ** 2 - RING_RI ** 2) * (RING_X[1] - RING_X[0]) * (1 - gap_angle / (2 * math.pi)) * PETG


def ring_mesh(n=72):
    """Points and faces of the split ring (a tube from just past the split round to just before it)."""
    gap = SPLIT_GAP / RING_RO
    a0, a1 = math.pi / 2 + gap / 2, math.pi / 2 + 2 * math.pi - gap / 2
    pts, faces = [], []
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        for x in RING_X:
            for r in (RING_RI, RING_RO):
                pts.append((x, r * math.cos(a), r * math.sin(a)))

    def v(i, xi, ri):
        return i * 4 + xi * 2 + ri
    for i in range(n):
        faces += [[v(i, 0, 1), v(i + 1, 0, 1), v(i + 1, 1, 1), v(i, 1, 1)],   # outer
                  [v(i, 1, 0), v(i + 1, 1, 0), v(i + 1, 0, 0), v(i, 0, 0)],   # inner
                  [v(i, 0, 0), v(i + 1, 0, 0), v(i + 1, 0, 1), v(i, 0, 1)],   # x0 face
                  [v(i, 1, 1), v(i + 1, 1, 1), v(i + 1, 1, 0), v(i, 1, 0)]]   # x1 face
    faces += [[v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0)], [v(n, 1, 0), v(n, 1, 1), v(n, 0, 1), v(n, 0, 0)]]
    return np.array(pts), faces


def box_points(c, R, size, n=7):
    """Points on the surface of a box (for clearance checks)."""
    g = np.linspace(-0.5, 0.5, n)
    pts = []
    for ax in range(3):
        for s in (-0.5, 0.5):
            for u in g:
                for v in g:
                    p = np.zeros(3)
                    p[ax] = s
                    o = [k for k in range(3) if k != ax]
                    p[o[0]], p[o[1]] = u, v
                    pts.append(p)
    return c + (np.array(pts) * np.array(size)) @ R.T


def look_at_orient(eye, target, up=Gf.Vec3d(0, 0, 1)):
    """Quaternion for a USD camera (looks down -Z, +Y up) at `eye` facing `target` (as in the build tool)."""
    z = (eye - target).GetNormalized()
    x = Gf.Cross(up, z).GetNormalized()
    y = Gf.Cross(z, x)
    m = Gf.Matrix3d(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
    return Gf.Quatf(m.ExtractRotation().GetQuat())


def _material(stage, path, rgb, roughness, metallic):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _matrix(R, c, scale=(1, 1, 1)):
    """USD (row-vector) matrix for rotation R (cols = local axes), scale and centre c, link06 frame."""
    M = np.eye(4)
    M[:3, :3] = R * np.array(scale)
    M[:3, 3] = np.asarray(c)
    return Gf.Matrix4d(*M.T.flatten())


def author(stage, say=print):
    """Ring clamp, arm, cradle, the D405 body (referenced asset), colliders, link06's mass, and the
    end-effector camera moved onto the D405's left imager with its intrinsics."""
    d = design()
    UsdGeom.Xform.Define(stage, HOLDER)
    black = _material(stage, f"{HOLDER}/Looks/petg", BLACK, 0.7, 0.0)
    metal = _material(stage, f"{HOLDER}/Looks/steel", METAL, 0.3, 0.9)
    added = []  # (mass, centre, size, R) in the link06 frame
    for name, c, R, size, collider, rgb, rho in d["boxes"]:
        cube = UsdGeom.Cube.Define(stage, f"{HOLDER}/{name}")
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube).AddTransformOp().Set(_matrix(R, c, size))
        if rgb is None:  # ring segments and the camera: collider only (their visuals are meshes)
            cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        else:
            cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(black)
        if collider:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if name == "camera_collider":
            added.append((CAM_MASS, np.asarray(c), np.array(size), R))
        elif rho is not None:
            added.append((size[0] * size[1] * size[2] * rho, np.asarray(c), np.array(size), R))
    pts, faces = ring_mesh()
    ring = UsdGeom.Mesh.Define(stage, f"{HOLDER}/ring")
    ring.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    ring.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
    ring.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    ring.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    ring.CreateDisplayColorAttr([Gf.Vec3f(*BLACK)])
    UsdShade.MaterialBindingAPI.Apply(ring.GetPrim()).Bind(black)
    m_ring = ring_mass()
    added.append((m_ring, np.array([np.mean(RING_X), 0.0, 0.0]),
                  np.array([RING_X[1] - RING_X[0], 2 * RING_RO, 2 * RING_RO]), np.eye(3)))
    bolt = UsdGeom.Cylinder.Define(stage, f"{HOLDER}/bolt")
    bolt.CreateRadiusAttr(BOLT_R)
    bolt.CreateHeightAttr(SPLIT_GAP + 2 * EAR[1] + 0.006)
    bolt.CreateAxisAttr("Y")
    UsdGeom.Xformable(bolt).AddTranslateOp().Set(Gf.Vec3d(float(np.mean(RING_X)), 0.0, RING_RO - 0.002 + EAR[2] / 2))
    UsdShade.MaterialBindingAPI.Apply(bolt.GetPrim()).Bind(metal)
    body = stage.DefinePrim(f"{HOLDER}/d405", "Xform")
    body.GetReferences().AddReference(D405_ASSET)
    UsdGeom.Xformable(body).AddTransformOp().Set(_matrix(d["R"], d["eye"]))

    # mass: the holder and the camera are clamped to link06
    ms = UsdPhysics.MassAPI(stage.GetPrimAtPath(LINK06))
    m0, c0 = ms.GetMassAttr().Get(), np.array(ms.GetCenterOfMassAttr().Get())
    i0 = np.array(ms.GetDiagonalInertiaAttr().Get())
    m_add = sum(a[0] for a in added)
    M_new = m0 + m_add
    c_new = (m0 * c0 + sum(a[0] * a[1] for a in added)) / M_new

    def shift(m, r):  # diagonal of the parallel-axis term
        return m * np.array([r[1] ** 2 + r[2] ** 2, r[0] ** 2 + r[2] ** 2, r[0] ** 2 + r[1] ** 2])
    inertia = i0 + shift(m0, c0 - c_new)
    for m, c, size, R in added:
        own = m / 12 * np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2])
        inertia += np.abs(R) @ own + shift(m, c - c_new)  # own inertia roughly rotated; the parts are small
    ms.GetMassAttr().Set(float(M_new))
    ms.GetCenterOfMassAttr().Set(Gf.Vec3f(*c_new))
    ms.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    say(f"[I] D405 ({CAM_MASS * 1000:.0f} g) on a ring clamp round link06 (ring, arm, cradle: "
        f"{(m_add - CAM_MASS) * 1000:.0f} g): link06 {m0:.3f} -> {M_new:.3f} kg, CoM {tuple(np.round(c0, 4))} -> "
        f"{tuple(np.round(c_new, 4))}, inertia {tuple(np.round(i0, 6))} -> {tuple(np.round(inertia, 6))}")

    # the camera: same prim path (ROS 2 graph, recorder), now on the left imager with D405 intrinsics
    cam = UsdGeom.Camera(stage.GetPrimAtPath(EE_CAMERA))
    eye = Gf.Vec3d(*d["eye"])
    target = Gf.Vec3d(*(d["eye"] + d["forward"]))
    xf = UsdGeom.Xformable(cam)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(eye)
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            q = look_at_orient(eye, target, Gf.Vec3d(*d["R"][:, 2]))
            op.Set(q if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd(q))
    cam.GetFocalLengthAttr().Set(FOCAL)
    cam.GetHorizontalApertureAttr().Set(APERTURE_H)
    cam.GetVerticalApertureAttr().Set(APERTURE_V)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1.0e7))  # RGB sees the fingers; depth < 7 cm is cut in the recorder
    say(f"[I] {EE_CAMERA} -> D405 left imager {tuple(np.round(d['eye'], 4))} (link06), "
        f"{math.degrees(YAW):.1f} deg toward the gripper, {-math.degrees(PITCH):.1f} deg down, focal {FOCAL:.2f} mm: "
        f"{HFOV:.0f} x {2 * math.degrees(math.atan(APERTURE_V / 2 / FOCAL)):.0f} deg at 16:9")
    return d


# ------------------------------------------------------------------ offline checks
def mesh_points(stage, root, frame=LINK06):
    xc = UsdGeom.XformCache()
    inv = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(frame)).GetInverse()
    out = []
    for q in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if q.IsA(UsdGeom.Mesh):
            m = xc.GetLocalToWorldTransform(q) * inv
            out += [m.Transform(Gf.Vec3d(*v)) for v in q.GetAttribute("points").Get()]
    return np.array(out)


def rotate_mover(P, angle):
    """Mover points at jointGripper = angle (rad), about the link06 y axis through PIVOT."""
    c, s = math.cos(angle), math.sin(angle)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (P - PIVOT) @ Ry.T + PIVOT


def min_dist(A, B, chunk=4000):
    best = np.inf
    for i in range(0, len(B), chunk):
        d = np.linalg.norm(A[:, None, :] - B[None, i:i + chunk, :], axis=2)
        best = min(best, d.min())
    return best


def holder_points(n=9):
    """Points on the holder's and camera's colliders (link06 frame)."""
    return np.vstack([box_points(c, R, s, n) for name, c, R, s, col, _, _ in design()["boxes"] if col])


def in_view(P):
    """Fraction of points P (link06) inside the camera's field of view, in front of the lens."""
    d = design()
    v = (P - d["eye"]) @ d["R"]  # camera frame: x forward, y left, z up
    ok = v[:, 0] > 0.003
    h = np.abs(np.arctan2(v[:, 1], v[:, 0])) < math.radians(HFOV / 2)
    vv = np.abs(np.arctan2(v[:, 2], v[:, 0])) < math.atan(APERTURE_V / 2 / FOCAL)
    return float((ok & h & vv).mean())


def check(stage, say=print):
    """Clearances of the holder's colliders to the moving jaw (whole range), the fixed jaw and link05 (whole
    wrist roll); the holder's place in the body frame at the authored (stowed) pose; what of it the camera sees."""
    H = holder_points()
    mover = mesh_points(stage, f"{MOVER}/collisions")[::7]
    worst = min(min_dist(H, rotate_mover(mover, a)) for a in np.linspace(0.0, -math.pi / 2, 19))
    stator = mesh_points(stage, f"{STATOR}/collisions")[::7]
    l05 = mesh_points(stage, f"{Z1}/link05/visuals")[::3]

    def roll(P, a):
        return np.c_[P[:, 0], P[:, 1] * math.cos(a) - P[:, 2] * math.sin(a), P[:, 1] * math.sin(a) + P[:, 2] * math.cos(a)]
    l5 = min(min_dist(roll(H, a), l05) for a in np.linspace(-math.pi, math.pi, 25))
    say(f"check: holder <-> moving jaw over 0..-90 deg: min {worst * 1000:.1f} mm; <-> fixed jaw "
        f"{min_dist(H, stator) * 1000:.1f} mm; <-> link05 over a full wrist roll {l5 * 1000:.1f} mm")
    xc = UsdGeom.XformCache()
    to_base = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(LINK06)) * \
        xc.GetLocalToWorldTransform(stage.GetPrimAtPath("/Unitree_B2W/base_link")).GetInverse()
    Hb = np.array([to_base.Transform(Gf.Vec3d(*p)) for p in H])
    say(f"check: holder in the body frame at the authored pose: x {Hb[:, 0].min():.3f}..{Hb[:, 0].max():.3f}, "
        f"y {Hb[:, 1].min():.3f}..{Hb[:, 1].max():.3f}, z {Hb[:, 2].min():.3f}..{Hb[:, 2].max():.3f}")
    seen = {name: round(in_view(box_points(c, R, s, 9)), 3) for name, c, R, s, col, _, _ in design()["boxes"]
            if not name.startswith(("camera", "lip", "cradle"))}
    seen = {k: v for k, v in seen.items() if v > 0}
    say(f"check: holder parts in the camera's view: {seen if seen else 'none'}")
    return min(worst, l5), Hb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=str(Path(__file__).resolve().parent.parent / "robots_usd/b2w_z1_gripper.usd"))
    args = ap.parse_args()
    check(Usd.Stage.Open(args.usd))


if __name__ == "__main__":
    sys.exit(main())
