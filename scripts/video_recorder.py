"""Multi-angle MuJoCo video recording with throw-point annotations."""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np


# OpenCV uses BGR for text, while MuJoCo uses RGBA for scene geometry.
POINT_STYLES = (
    ("Theoretical release", "theoretical_release", (0.10, 0.85, 1.00, 1.0), (255, 215, 40)),
    ("Actual release", "actual_release", (1.00, 0.25, 0.20, 1.0), (45, 60, 255)),
    ("Theoretical landing", "theoretical_landing", (0.25, 1.00, 0.25, 1.0), (70, 230, 70)),
    ("Actual landing", "actual_landing", (0.85, 0.25, 1.00, 1.0), (230, 70, 220)),
)


CAMERA_PRESETS = {
    "front": dict(azimuth=90.0, elevation=-17.0, distance=2.15),
    "side": dict(azimuth=180.0, elevation=-14.0, distance=2.05),
    "oblique": dict(azimuth=135.0, elevation=-22.0, distance=2.25),
    "top": dict(azimuth=90.0, elevation=-70.0, distance=2.35),
    # User-supplied MuJoCo camera pose:
    # <camera pos="-0.435 2.678 1.346"
    #         xyaxes="-0.999 0.052 -0.000 -0.022 -0.429 0.903"/>
    "user_pose": dict(
        pos=(-0.435, 2.678, 1.346),
        xyaxes=(-0.999, 0.052, -0.000, -0.022, -0.429, 0.903),
    ),
}


def add_throw_markers(scene, points: dict):
    """Add every currently available throw point to an MjvScene."""
    for _, key, rgba, _ in POINT_STYLES:
        point = points.get(key)
        if point is None or np.asarray(point).shape != (3,) or not np.all(np.isfinite(point)):
            continue
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.022, 0.022, 0.022]),
            np.asarray(point, dtype=float),
            np.eye(3).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


class ThrowVideoRecorder:
    """Record synchronized views and draw persistent world-space markers."""

    def __init__(
        self,
        model: mujoco.MjModel,
        output_dir: str,
        camera_names: list[str],
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
    ):
        unknown = sorted(set(camera_names) - set(CAMERA_PRESETS))
        if unknown:
            raise ValueError(
                f"Unknown camera(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(CAMERA_PRESETS)}"
            )
        self.model = model
        self.fps = float(fps)
        self.frame_period = 1.0 / self.fps
        self.next_frame_time = 0.0
        self.width, self.height = int(width), int(height)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # The XML's default off-screen framebuffer is only 640x480.  MuJoCo's
        # Renderer refuses larger images unless these limits are raised before
        # its OpenGL context is created.
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
        # Camera-following fill light for clear robot detail from every view.
        # The XML has one top-down directional light, which leaves the sides
        # of the Frankas too dark from low/front camera poses.
        model.vis.headlight.active = 1
        model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
        model.vis.headlight.diffuse[:] = [0.80, 0.80, 0.80]
        model.vis.headlight.specular[:] = [0.35, 0.35, 0.35]
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.cameras = {
            name: (self._make_camera(CAMERA_PRESETS[name]), CAMERA_PRESETS[name])
            for name in camera_names
        }
        self.writers = {}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for name in camera_names:
            path = self.output_dir / f"throw_{name}.mp4"
            writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self.width, self.height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
            self.writers[name] = writer

    @staticmethod
    def _make_camera(spec: dict):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        if "pos" in spec:
            # A close free-camera equivalent is needed only to initialize the
            # scene; the exact position/orientation is applied before render.
            pos = np.asarray(spec["pos"], dtype=float)
            xyaxes = np.asarray(spec["xyaxes"], dtype=float).reshape(2, 3)
            forward = -np.cross(xyaxes[0], xyaxes[1])
            forward /= np.linalg.norm(forward)
            distance = 3.81
            camera.lookat[:] = pos + distance * forward
            camera.distance = distance
        else:
            camera.lookat[:] = [-0.42, 0.42, 0.35]
            camera.azimuth = spec["azimuth"]
            camera.elevation = spec["elevation"]
            camera.distance = spec["distance"]
        return camera

    def _apply_exact_pose(self, spec: dict):
        if "pos" not in spec:
            return
        pos = np.asarray(spec["pos"], dtype=float)
        xyaxes = np.asarray(spec["xyaxes"], dtype=float).reshape(2, 3)
        forward = -np.cross(xyaxes[0], xyaxes[1])
        forward /= np.linalg.norm(forward)
        up = xyaxes[1] / np.linalg.norm(xyaxes[1])
        # Both OpenGL cameras are set because MuJoCo stores a stereo pair even
        # when rendering a normal mono frame.
        for gl_camera in self.renderer.scene.camera:
            gl_camera.pos[:] = pos
            gl_camera.forward[:] = forward
            gl_camera.up[:] = up

    @staticmethod
    def _valid(point) -> bool:
        return point is not None and np.asarray(point).shape == (3,) and np.all(np.isfinite(point))

    def _annotate_frame(self, rgb, t: float, phase: str, points: dict):
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 14), (385, 166), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, frame)
        cv2.putText(frame, f"t = {t:5.2f} s   {phase}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y = 67
        for label, key, _, bgr in POINT_STYLES:
            available = self._valid(points.get(key))
            color = bgr if available else (115, 115, 115)
            cv2.circle(frame, (34, y - 5), 7, color, -1, cv2.LINE_AA)
            suffix = "" if available else " (pending)"
            cv2.putText(frame, label + suffix, (52, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, color, 1, cv2.LINE_AA)
            y += 27
        return frame

    def capture(self, data: mujoco.MjData, t: float, phase: str, points: dict):
        if t + 1e-9 < self.next_frame_time:
            return
        # Avoid cumulative timing drift if simulation dt is not a divisor of FPS.
        self.next_frame_time += self.frame_period
        for name, (camera, spec) in self.cameras.items():
            self.renderer.update_scene(data, camera=camera)
            add_throw_markers(self.renderer.scene, points)
            self._apply_exact_pose(spec)
            rgb = self.renderer.render()
            self.writers[name].write(self._annotate_frame(rgb, t, phase, points))

    @property
    def paths(self):
        return [self.output_dir / f"throw_{name}.mp4" for name in self.cameras]

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.renderer.close()
