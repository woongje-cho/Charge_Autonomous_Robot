"""
Bridge Guardian — Sensor Fusion
Fuses camera (YOLOv8 person detections) with 2D LiDAR scans to produce
distance-validated person detections with angular matching.

Camera-to-LiDAR mapping:
  pixel_x  →  angle_deg = (x / frame_width - 0.5) * camera_hfov
  where camera_hfov defaults to 69° (Intel RealSense D435 horizontal FOV).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FusedPerson dataclass
# ---------------------------------------------------------------------------
@dataclass
class FusedPerson:
    """A person detection enriched with fused sensor data."""
    track_id: Optional[int]             # from BoT-SORT tracker (None if unavailable)
    bbox: np.ndarray                    # [x1, y1, x2, y2] pixels
    center: Tuple[float, float]         # (cx, cy) pixel centre
    confidence: float                   # YOLO detection confidence
    camera_angle_deg: float             # estimated angle from camera centre
    lidar_distance_m: Optional[float]   # matched LiDAR distance (None if no match)
    depth_distance_m: Optional[float]   # RealSense depth distance (None if unavailable)
    fused_distance_m: Optional[float]   # best available distance estimate
    lidar_matched: bool                 # True if LiDAR detection was matched
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# SensorFusion
# ---------------------------------------------------------------------------
class SensorFusion:
    """Fuses camera person detections with 2D LiDAR scan data.

    The fusion process:
    1. Map each camera detection's pixel x-coordinate to an angle via the
       camera's horizontal field of view.
    2. Query the LiDAR scan for nearby detections within an angular tolerance.
    3. Produce a :class:`FusedPerson` with the best available distance.

    Parameters
    ----------
    config : dict
        Optional keys:
        - ``camera_hfov_deg``       (float) – camera horizontal FOV (default 69.0°)
        - ``angle_tolerance_deg``   (float) – max angular mismatch for matching (default 10.0°)
        - ``distance_tolerance_m``  (float) – max depth vs lidar discrepancy (default 1.5)
        - ``lidar_sector_width_deg``(float) – sector width for nearest query (default 15.0°)
        - ``frame_width``           (int)   – camera frame width in pixels (default 1280)
    """

    def __init__(self, config: dict) -> None:
        self._hfov: float = config.get("camera_hfov_deg", 69.0)
        self._angle_tol: float = config.get("angle_tolerance_deg", 10.0)
        self._dist_tol: float = config.get("distance_tolerance_m", 1.5)
        self._sector_width: float = config.get("lidar_sector_width_deg", 15.0)
        self._frame_width: int = config.get("frame_width", 1280)

        logger.info(
            "SensorFusion initialised (hfov=%.1f°, angle_tol=%.1f°, "
            "sector_width=%.1f°, frame_width=%d)",
            self._hfov, self._angle_tol, self._sector_width, self._frame_width,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def camera_hfov_deg(self) -> float:
        return self._hfov

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @frame_width.setter
    def frame_width(self, value: int) -> None:
        """Allow runtime update when frame resolution changes."""
        if value > 0:
            self._frame_width = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pixel_x_to_angle(self, pixel_x: float) -> float:
        """Convert a pixel x-coordinate to an angle relative to camera centre.

        Returns angle in degrees where 0° = camera centre, positive = right.
        The returned angle is then mapped to the LiDAR frame where the
        camera forward direction corresponds to LiDAR 0° (configurable).

        Formula: angle = (pixel_x / frame_width - 0.5) * hfov
        """
        return (pixel_x / self._frame_width - 0.5) * self._hfov

    def fuse(
        self,
        detections: list,
        lidar_handler=None,
        depth_frame=None,
        camera_handler=None,
        lidar_forward_deg: float = 0.0,
    ) -> List[FusedPerson]:
        """Fuse camera person detections with LiDAR data.

        Parameters
        ----------
        detections : List[DetectedPerson]
            Output from :meth:`PersonDetector.detect`.
        lidar_handler : LidarHandler, optional
            Active LiDAR handler for scan queries.
        depth_frame : optional
            RealSense depth frame for depth-based distance.
        camera_handler : optional
            Camera handler with ``get_median_depth_in_bbox`` method.
        lidar_forward_deg : float
            The LiDAR angle (in degrees) that corresponds to the camera's
            forward direction. Default 0° assumes camera and LiDAR face
            the same way.

        Returns
        -------
        List[FusedPerson]
            Fused detections with best-available distance estimates.
        """
        # Get LiDAR scan and detections if handler available
        lidar_scan = None
        lidar_detections = None
        if lidar_handler is not None:
            lidar_scan = lidar_handler.get_scan()
            if lidar_scan.valid:
                lidar_detections = lidar_handler.detect_objects(lidar_scan)

        fused_persons: List[FusedPerson] = []

        for det in detections:
            cx, cy = det.center
            camera_angle = self.pixel_x_to_angle(cx)

            # Map camera angle to LiDAR coordinate frame
            lidar_angle = (lidar_forward_deg + camera_angle) % 360.0

            # --- LiDAR matching ---
            lidar_dist: Optional[float] = None
            lidar_matched = False

            if lidar_handler is not None and lidar_scan is not None and lidar_scan.valid:
                # Strategy 1: match against detected clusters
                if lidar_detections:
                    best_match = self._match_lidar_detection(
                        lidar_angle, lidar_detections,
                    )
                    if best_match is not None:
                        lidar_dist = best_match.distance_m
                        lidar_matched = True

                # Strategy 2: fallback to nearest point in sector
                if lidar_dist is None:
                    sector_dist = lidar_handler.get_nearest_in_sector(
                        lidar_angle, self._sector_width, lidar_scan,
                    )
                    if sector_dist is not None:
                        lidar_dist = sector_dist
                        lidar_matched = True

            # --- Depth camera distance ---
            depth_dist: Optional[float] = None
            if depth_frame is not None and camera_handler is not None:
                x1, y1, x2, y2 = det.bbox.astype(int)
                d = camera_handler.get_median_depth_in_bbox(
                    depth_frame, int(x1), int(y1), int(x2), int(y2),
                )
                if d > 0.0:
                    depth_dist = d

            # --- Best distance estimate (graceful degradation) ---
            fused_dist = self._compute_fused_distance(lidar_dist, depth_dist)

            fused_persons.append(FusedPerson(
                track_id=det.track_id,
                bbox=det.bbox,
                center=det.center,
                confidence=det.confidence,
                camera_angle_deg=camera_angle,
                lidar_distance_m=lidar_dist,
                depth_distance_m=depth_dist,
                fused_distance_m=fused_dist,
                lidar_matched=lidar_matched,
                timestamp=time.time(),
            ))

        return fused_persons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_lidar_detection(
        self,
        target_angle: float,
        lidar_detections: list,
    ) -> Optional[object]:
        """Find the best matching LiDAR detection by angular proximity.

        Returns the LidarDetection with the smallest angular difference
        within ``angle_tolerance_deg``, or None.
        """
        best = None
        best_diff = self._angle_tol

        for ld in lidar_detections:
            diff = abs(self._angle_diff(target_angle, ld.angle_deg))
            if diff < best_diff:
                best_diff = diff
                best = ld

        return best

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed shortest angular difference in degrees [-180, 180]."""
        d = (b - a) % 360.0
        if d > 180.0:
            d -= 360.0
        return d

    def _compute_fused_distance(
        self,
        lidar_dist: Optional[float],
        depth_dist: Optional[float],
    ) -> Optional[float]:
        """Compute the best distance estimate from available sensors.

        Priority order:
        1. Both available & consistent → weighted average (LiDAR 60%, depth 40%)
        2. Both available & inconsistent → prefer LiDAR (more reliable for 2D range)
        3. Only one available → use that
        4. Neither → None
        """
        if lidar_dist is not None and depth_dist is not None:
            if abs(lidar_dist - depth_dist) <= self._dist_tol:
                # Consistent — weighted average
                return lidar_dist * 0.6 + depth_dist * 0.4
            else:
                # Inconsistent — trust LiDAR for planar distance
                logger.debug(
                    "Sensor distance mismatch: lidar=%.2fm, depth=%.2fm — using LiDAR",
                    lidar_dist, depth_dist,
                )
                return lidar_dist

        if lidar_dist is not None:
            return lidar_dist
        if depth_dist is not None:
            return depth_dist

        return None
