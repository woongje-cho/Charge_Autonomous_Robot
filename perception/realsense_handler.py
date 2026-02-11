"""
Intel RealSense D435 camera handler with video file fallback.

Supports two modes:
  - "realsense": Live capture from Intel RealSense D435 with aligned depth.
  - Any other string: Treated as a video file path (depth unavailable).
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# Graceful pyrealsense2 import
_RS_AVAILABLE = False
try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    rs = None  # type: ignore
    logger.warning("pyrealsense2 not installed — RealSense mode unavailable")


@dataclass
class FrameData:
    """Container for a single synchronized frame pair."""
    color_image: np.ndarray              # BGR, shape (H, W, 3)
    depth_frame: Optional[Any]           # rs.depth_frame or None (video mode)
    depth_image: Optional[np.ndarray]    # uint16 depth map or None
    timestamp: float                     # time.time()


class CameraHandler:
    """Dual-mode camera handler: Intel RealSense D435 live capture or video file playback."""

    def __init__(self, config: dict) -> None:
        self._source: str = config.get("source", "realsense")
        self._width: int = config.get("width", 1280)
        self._height: int = config.get("height", 720)
        self._fps: int = config.get("fps", 30)
        self._depth_width: int = config.get("depth_width", 1280)
        self._depth_height: int = config.get("depth_height", 720)

        self._pipeline = None
        self._capture: Optional[cv2.VideoCapture] = None
        self.align = None
        self.color_intrinsics = None
        self.depth_scale: float = 0.0
        self._stopped: bool = False
        self._last_frame_time: float = 0.0

        if self._source == "realsense":
            self._init_realsense()
        else:
            self._init_video(self._source)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_realsense(self) -> bool:
        """True when operating in live RealSense mode."""
        return self._source == "realsense"

    @property
    def fps(self) -> int:
        return self._fps

    # ------------------------------------------------------------------
    # RealSense initialisation
    # ------------------------------------------------------------------

    def _init_realsense(self) -> None:
        if not _RS_AVAILABLE:
            raise RuntimeError(
                "RealSense mode requested but pyrealsense2 is not installed. "
                "Install with: pip install pyrealsense2"
            )

        logger.info("Initialising RealSense pipeline (%dx%d @ %d fps)", self._width, self._height, self._fps)

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self._depth_width, self._depth_height, rs.format.z16, self._fps)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        profile = self._pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        # Cache intrinsics and depth scale
        self._init_intrinsics(profile)

        # Warm-up: discard first 5 frames to let auto-exposure settle
        logger.info("Warming up — discarding 5 frames")
        for _ in range(5):
            self._pipeline.wait_for_frames(5000)

        logger.info("RealSense pipeline ready (depth_scale=%.6f)", self.depth_scale)

    def _init_intrinsics(self, profile) -> None:
        """Cache color intrinsics and depth sensor scale from the active profile."""
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Retrieve colour stream intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_stream.get_intrinsics()

        logger.debug(
            "Color intrinsics: %dx%d, fx=%.2f, fy=%.2f",
            self.color_intrinsics.width,
            self.color_intrinsics.height,
            self.color_intrinsics.fx,
            self.color_intrinsics.fy,
        )

    # ------------------------------------------------------------------
    # Video file initialisation
    # ------------------------------------------------------------------

    def _init_video(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        logger.info("Opening video file: %s", path)
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open: {path}")

        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Video opened — %.1f fps, %d frames", actual_fps, frame_count)

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def get_frames(self) -> Optional[FrameData]:
        """Return the next synchronised frame pair, or None on failure / end-of-stream."""
        if self._stopped:
            return None

        if self.is_realsense:
            return self._get_realsense_frames()
        return self._get_video_frame()

    def _get_realsense_frames(self) -> Optional[FrameData]:
        try:
            frames = self._pipeline.wait_for_frames(5000)
        except Exception as exc:
            logger.error("RealSense wait_for_frames failed: %s", exc)
            return None

        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            logger.warning("Incomplete frameset — skipping")
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

        return FrameData(
            color_image=color_image,
            depth_frame=depth_frame,
            depth_image=depth_image,
            timestamp=time.time(),
        )

    def _get_video_frame(self) -> Optional[FrameData]:
        # FPS throttling
        now = time.time()
        elapsed = now - self._last_frame_time
        target_interval = 1.0 / self._fps
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)

        ret, frame = self._capture.read()
        self._last_frame_time = time.time()

        if not ret:
            logger.info("Video stream ended")
            return None

        # Resize to configured resolution if needed
        h, w = frame.shape[:2]
        if w != self._width or h != self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        return FrameData(
            color_image=frame,
            depth_frame=None,
            depth_image=None,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Depth utilities
    # ------------------------------------------------------------------

    def get_distance(self, depth_frame: Any, x: int, y: int) -> float:
        """Return depth at (x, y) in metres.  Returns 0.0 when depth is unavailable."""
        if depth_frame is None:
            return 0.0

        # Clamp to frame bounds
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        return depth_frame.get_distance(x, y)

    def pixel_to_3d(
        self, depth_frame: Any, px: int, py: int
    ) -> Optional[Tuple[float, float, float]]:
        """Deproject a pixel (px, py) to a 3-D point (X, Y, Z) in metres.

        Returns None when depth data or intrinsics are unavailable.
        """
        if depth_frame is None or self.color_intrinsics is None:
            return None

        depth = self.get_distance(depth_frame, px, py)
        if depth <= 0.0:
            return None

        point_3d = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics, [float(px), float(py)], depth
        )
        return (point_3d[0], point_3d[1], point_3d[2])

    def get_median_depth_in_bbox(
        self, depth_frame: Any, x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """Return the median non-zero depth (metres) from a ~20-point grid inside the bbox.

        Returns 0.0 when depth data is unavailable or all samples are zero.
        """
        if depth_frame is None:
            return 0.0

        w = depth_frame.get_width()
        h = depth_frame.get_height()

        # Clamp bbox to frame bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Build a grid of roughly 20 sample points (5 cols × 4 rows)
        num_cols = 5
        num_rows = 4
        xs = np.linspace(x1, x2, num_cols, dtype=int)
        ys = np.linspace(y1, y2, num_rows, dtype=int)

        depths = []
        for sy in ys:
            for sx in xs:
                d = depth_frame.get_distance(int(sx), int(sy))
                if d > 0.0:
                    depths.append(d)

        if not depths:
            return 0.0

        return float(np.median(depths))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Release camera / pipeline resources."""
        if self._stopped:
            return
        self._stopped = True

        if self.is_realsense and self._pipeline is not None:
            try:
                self._pipeline.stop()
                logger.info("RealSense pipeline stopped")
            except Exception as exc:
                logger.warning("Error stopping RealSense pipeline: %s", exc)
        elif self._capture is not None:
            self._capture.release()
            logger.info("Video capture released")
