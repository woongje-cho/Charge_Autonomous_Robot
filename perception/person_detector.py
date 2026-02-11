"""YOLOv8n-pose person detection with built-in BoT-SORT tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# COCO 17 keypoint indices
# ---------------------------------------------------------------------------
KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (KP_NOSE, KP_L_EYE), (KP_NOSE, KP_R_EYE),
    (KP_L_EYE, KP_L_EAR), (KP_R_EYE, KP_R_EAR),
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP), (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE), (KP_L_KNEE, KP_L_ANKLE),
    (KP_R_HIP, KP_R_KNEE), (KP_R_KNEE, KP_R_ANKLE),
]


# ---------------------------------------------------------------------------
# DetectedPerson dataclass
# ---------------------------------------------------------------------------
@dataclass
class DetectedPerson:
    """Single detected person with pose keypoints and optional track ID."""

    bbox: np.ndarray                # [x1, y1, x2, y2] pixels
    center: Tuple[float, float]     # (cx, cy) centre of bbox
    keypoints: np.ndarray           # (17, 2) x,y pixel coordinates
    kp_conf: np.ndarray             # (17,) per-keypoint confidence
    confidence: float               # detection confidence
    track_id: Optional[int]         # persistent track ID from BoT-SORT (None if tracking unavailable)


# ---------------------------------------------------------------------------
# PersonDetector
# ---------------------------------------------------------------------------
class PersonDetector:
    """YOLOv8n-pose detector with integrated BoT-SORT tracking.

    Parameters
    ----------
    config : dict
        Required keys:
        - model_path      (str)       – path to YOLOv8 pose weights
        - confidence       (float)     – detection confidence threshold
        - iou_threshold    (float)     – NMS IoU threshold
        - device           (int | str) – CUDA device ordinal or ``"cpu"``
        - input_size       (int)       – square input dimension for warm-up
        - tracker_config   (str)       – path to BoT-SORT tracker YAML
    """

    def __init__(self, config: dict) -> None:
        self.conf: float = config["confidence"]
        self.iou: float = config["iou_threshold"]
        self.device = config["device"]
        self.tracker_config: str = config["tracker_config"]

        logger.info(
            "Loading YOLO pose model from %s (device=%s, conf=%.2f, iou=%.2f)",
            config["model_path"],
            self.device,
            self.conf,
            self.iou,
        )
        self.model = YOLO(config["model_path"])

        # Warm-up: trigger JIT / TensorRT compilation on a dummy frame
        input_size: int = config["input_size"]
        dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        logger.info("Warming up model with %dx%d dummy frame …", input_size, input_size)
        self.model(dummy, verbose=False)
        logger.info("Model warm-up complete.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[DetectedPerson]:
        """Run detection **with** BoT-SORT tracking on *frame*.

        Returns a list of :class:`DetectedPerson` instances.  Track IDs are
        ``None`` when the tracker has not yet assigned an identity.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            device=self.device,
            tracker=self.tracker_config,
            verbose=False,
        )
        return self._parse_results(results, tracked=True)

    def detect_no_track(self, frame: np.ndarray) -> List[DetectedPerson]:
        """Run detection **without** tracking (single-frame analysis).

        All ``track_id`` fields will be ``None``.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results, tracked=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(results, *, tracked: bool) -> List[DetectedPerson]:
        """Convert raw YOLO results into a list of :class:`DetectedPerson`.

        Parameters
        ----------
        results
            Output from ``model.track(...)`` or ``model(...)``.
        tracked : bool
            If ``True`` we attempt to read track IDs from ``boxes.id``.
        """
        # --- edge-case guards -------------------------------------------
        if results is None or len(results) == 0:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        # Pose keypoints are mandatory for our pipeline
        if result.keypoints is None:
            return []

        # --- extract tensors → numpy ------------------------------------
        boxes_xyxy: np.ndarray = result.boxes.xyxy.cpu().numpy()       # (N, 4)
        confs: np.ndarray = result.boxes.conf.cpu().numpy()            # (N,)
        kps_xy: np.ndarray = result.keypoints.xy.cpu().numpy()         # (N, 17, 2)
        kps_conf: np.ndarray = result.keypoints.conf.cpu().numpy()     # (N, 17)

        # Track IDs may be unavailable even in tracking mode
        track_ids: Optional[np.ndarray] = None
        if tracked and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)      # (N,)

        # --- build DetectedPerson list ----------------------------------
        n = boxes_xyxy.shape[0]
        persons: List[DetectedPerson] = []
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i]
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            tid: Optional[int] = None
            if track_ids is not None:
                tid = int(track_ids[i])

            persons.append(
                DetectedPerson(
                    bbox=boxes_xyxy[i],
                    center=(cx, cy),
                    keypoints=kps_xy[i],
                    kp_conf=kps_conf[i],
                    confidence=float(confs[i]),
                    track_id=tid,
                )
            )

        return persons
