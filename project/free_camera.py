"""Let the viewer camera be moved by hand during a streamed run.

WHY THE CAMERA FEELS STUCK
==========================
With `viewer.origin_type = "asset_root"` Isaac Lab re-points the camera at the
tracked asset on EVERY rendering step -- the code says so itself:
"for assets, the camera view is updated at each rendering step"
(isaaclab/envs/ui/viewport_camera_controller.py:127). So a mouse drag is undone a
frame later and the camera looks frozen on the robot. Nothing is broken; the
follow mode is simply fighting you.

Switching `origin_type` back to "world" stops those per-frame updates and the
viewport behaves like any normal Isaac Sim camera again (left drag orbit, middle
drag pan, right drag / scroll zoom, F to frame the selection in the GUI).

Both modes are useful -- following is what you want while the robot drives away,
free is what you want to inspect a pose -- so this switches between them at
runtime instead of making you restart:

    C   snap the camera to the robot ONCE, then leave it free
    V   toggle follow on/off

C rather than F, because F is already "frame selected" in the Isaac viewport.

    from free_camera import CameraToggle
    cam = CameraToggle(env.unwrapped, following=args.follow)

Safe when there is no viewport at all (headless without a livestream): it simply
does nothing.
"""

from __future__ import annotations


class CameraToggle:
    """Runtime switch between a robot-following camera and a free one."""

    def __init__(self, env, asset: str = "robot", offset=(2.5, 2.5, 1.8), following: bool = False):
        self.env = env
        self.asset = asset
        self.offset = offset
        self.following = following
        self.ctrl = getattr(env, "viewport_camera_controller", None)
        self._sub = None
        if self.ctrl is None:
            return  # headless: no viewport to control
        self._bind_keys()
        self.apply()

    # ------------------------------------------------------------------ keys
    def _bind_keys(self):
        try:
            import carb
            import omni.appwindow

            self._carb = carb
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._input = carb.input.acquire_input_interface()
            self._sub = self._input.subscribe_to_keyboard_events(keyboard, self._on_key)
            print("CAM| keys: C = snap to robot (stays free), V = toggle follow", flush=True)
        except Exception as exc:  # no input backend -- not fatal, just no hotkeys
            print(f"CAM| hotkeys unavailable ({exc}); camera mode fixed at start-up", flush=True)

    def _on_key(self, event, *_):
        try:
            if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
                return True
            if event.input == self._carb.input.KeyboardInput.V:
                self.following = not self.following
                self.apply()
                print(f"CAM| follow {'ON' if self.following else 'OFF (camera is yours)'}", flush=True)
            elif event.input == self._carb.input.KeyboardInput.C:
                self.following = False
                self.snap()
                print("CAM| snapped to the robot; camera is free", flush=True)
        except Exception as exc:
            print(f"CAM| key handling failed: {exc}", flush=True)
        return True

    # ------------------------------------------------------------------ view
    def _robot_pos(self):
        try:
            return self.env.scene[self.asset].data.root_pos_w[0].cpu().tolist()
        except Exception:
            return [0.0, 0.0, 0.0]

    def snap(self):
        """Place the camera near the robot once, then leave it alone."""
        pos = self._robot_pos()
        self.ctrl.update_view_to_world()          # stops the per-frame re-pointing
        self.ctrl.update_view_location(
            eye=[pos[0] + self.offset[0], pos[1] + self.offset[1], pos[2] + self.offset[2]],
            lookat=[pos[0], pos[1], pos[2] + 0.2],
        )

    def apply(self):
        if self.following:
            self.ctrl.update_view_to_asset_root(self.asset)
        else:
            self.snap()

    def free(self):
        """Stop the per-frame tracking. Also the shutdown fix: Isaac Lab's
        tracking callback keeps firing while the app tears down, and in
        asset_root mode it dereferences env.scene after the env is gone --
        'ManagerBasedRLEnv has no attribute scene'. In world mode the callback
        does nothing (viewport_camera_controller.py:226)."""
        if self.ctrl is not None:
            try:
                self.ctrl.update_view_to_world()
            except Exception:
                pass
        self.following = False

    def close(self):
        self.free()
        if self._sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(
                    __import__("omni.appwindow", fromlist=["x"]).get_default_app_window().get_keyboard(),
                    self._sub,
                )
            except Exception:
                pass
            self._sub = None
