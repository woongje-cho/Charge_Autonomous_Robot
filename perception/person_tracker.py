"""Multi-person temporal state tracker using YOLOv8 BoT-SORT track IDs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TrackedPerson dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrackedPerson:
    """Persistent state for a single tracked individual across frames."""

    track_id: int
    bbox: np.ndarray                           # latest [x1, y1, x2, y2]
    center: Tuple[float, float]                # latest (cx, cy)
    keypoints: np.ndarray                      # latest (17, 2)
    kp_conf: np.ndarray                        # latest (17,)
    confidence: float                          # latest detection confidence
    positions: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=150),
    )  # (cx, cy, timestamp)
    keypoints_history: Deque[Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=30),
    )  # (keypoints, kp_conf)
    first_seen: float = 0.0
    last_seen: float = 0.0
    depth_m: Optional[float] = None
    frame_count: int = 0


# ---------------------------------------------------------------------------
# PersonTracker
# ---------------------------------------------------------------------------
class PersonTracker:
    """Maintains per-person temporal history keyed by BoT-SORT track IDs.

    This class does **not** perform tracking itself — it consumes the
    ``track_id`` already assigned by YOLOv8's BoT-SORT tracker inside
    :class:`perception.person_detector.PersonDetector` and accumulates
    position / keypoint history for downstream analysis (behaviour,
    loitering, fall detection, etc.).

    Parameters
    ----------
    config : dict
        Required keys:
        - ``max_age_seconds``       (float) – drop track after this silence
        - ``position_history_size`` (int)   – maxlen for positions deque
        - ``keypoint_history_size`` (int)   – maxlen for keypoints_history deque
    """

    def __init__(self, config: dict) -> None:
        self._max_age: float = float(config["max_age_seconds"])
        self._pos_maxlen: int = int(config["position_history_size"])
        self._kp_maxlen: int = int(config["keypoint_history_size"])
        self._tracks: Dict[int, TrackedPerson] = {}

        logger.info(
            "PersonTracker initialised (max_age=%.1fs, pos_hist=%d, kp_hist=%d)",
            self._max_age,
            self._pos_maxlen,
            self._kp_maxlen,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list,
        timestamp: float,
        depth_frame=None,
        camera_handler=None,
    ) -> List[TrackedPerson]:
        """Ingest new detections and return all active tracks.

        Parameters
        ----------
        detections : List[DetectedPerson]
            Output from :meth:`PersonDetector.detect`.  Entries without a
            ``track_id`` (i.e. ``track_id is None``) are silently skipped.
        timestamp : float
            Monotonic or wall-clock timestamp for this frame.
        depth_frame : optional
            RealSense depth frame for depth queries.  May be ``None``.
        camera_handler : optional
            :class:`RealsenseHandler` instance exposing
            ``get_median_depth_in_bbox``.  Required together with
            *depth_frame* for depth estimation.

        Returns
        -------
        List[TrackedPerson]
            All currently active tracked persons.
        """
        # Import here to avoid circular dependency with person_detector
        from perception.person_detector import DetectedPerson  # noqa: F811

        for det in detections:
            if det.track_id is None:
                continue

            tid: int = det.track_id

            if tid not in self._tracks:
                # --- new track ------------------------------------------------
                person = TrackedPerson(
                    track_id=tid,
                    bbox=det.bbox,
                    center=det.center,
                    keypoints=det.keypoints,
                    kp_conf=det.kp_conf,
                    confidence=det.confidence,
                    positions=deque(maxlen=self._pos_maxlen),
                    keypoints_history=deque(maxlen=self._kp_maxlen),
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frame_count=0,
                )
                self._tracks[tid] = person
                logger.debug("New track id=%d at (%.0f, %.0f)", tid, det.center[0], det.center[1])
            else:
                # --- existing track -------------------------------------------
                person = self._tracks[tid]
                person.bbox = det.bbox
                person.center = det.center
                person.keypoints = det.keypoints
                person.kp_conf = det.kp_conf
                person.confidence = det.confidence
                person.last_seen = timestamp

            # Append history entries
            person.positions.append((det.center[0], det.center[1], timestamp))
            person.keypoints_history.append((det.keypoints.copy(), det.kp_conf.copy()))
            person.frame_count += 1

            # Optional depth estimation
            if depth_frame is not None and camera_handler is not None:
                x1, y1, x2, y2 = det.bbox.astype(int)
                depth_val = camera_handler.get_median_depth_in_bbox(
                    depth_frame, int(x1), int(y1), int(x2), int(y2),
                )
                if depth_val > 0.0:
                    person.depth_m = depth_val

        self._cleanup_stale(timestamp)
        return self.get_active_persons()

    def get_person(self, track_id: int) -> Optional[TrackedPerson]:
        """Return the :class:`TrackedPerson` for *track_id*, or ``None``."""
        return self._tracks.get(track_id)

    def get_active_persons(self) -> List[TrackedPerson]:
        """Return a list of all currently active tracked persons."""
        return list(self._tracks.values())

    def get_track_count(self) -> int:
        """Return the number of currently active tracks."""
        return len(self._tracks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_stale(self, current_timestamp: float) -> None:
        """Remove tracks not seen for longer than ``max_age_seconds``."""
        stale_ids = [
            tid
            for tid, person in self._tracks.items()
            if current_timestamp - person.last_seen > self._max_age
        ]
        for tid in stale_ids:
            person = self._tracks.pop(tid)
            logger.info(
                "Removed stale track id=%d (last_seen=%.2fs ago, lived %.1fs, %d frames)",
                tid,
                current_timestamp - person.last_seen,
                person.last_seen - person.first_seen,
                person.frame_count,
            )
