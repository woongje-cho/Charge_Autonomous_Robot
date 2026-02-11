"""
ChargeMate - 손 감지 및 손 흔들기 인식 모듈
MediaPipe Hands를 사용하여 손 흔드는 사용자를 감지한다.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graceful Import
# ---------------------------------------------------------------------------

_MEDIAPIPE_AVAILABLE = False

try:
    import mediapipe as mp

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    logger.warning("MediaPipe 사용 불가 - 손 감지 비활성화")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class HandInfo:
    """감지된 손 정보"""

    landmarks: np.ndarray  # 21개 랜드마크 (x, y, z)
    handedness: str  # "Left" | "Right"
    center: Tuple[float, float]  # 손 중심 (x, y) 정규화
    bbox: Tuple[int, int, int, int]  # 바운딩 박스 (x1, y1, x2, y2)
    confidence: float


@dataclass
class WavingPerson:
    """손 흔드는 사람 정보"""

    track_id: Optional[int]  # 사람 추적 ID
    center_x: float  # 프레임 내 x 좌표
    center_y: float  # 프레임 내 y 좌표
    distance_m: Optional[float]  # 거리 (미터)
    wave_count: int  # 손 흔든 횟수
    last_wave_time: float  # 마지막 손 흔들기 시간
    bbox: Tuple[int, int, int, int]  # 사람 바운딩 박스


# ---------------------------------------------------------------------------
# Hand Detector
# ---------------------------------------------------------------------------


class HandDetector:
    """
    MediaPipe 기반 손 감지 및 손 흔들기 인식기.

    Parameters
    ----------
    config : dict
        hand_detection 설정 딕셔너리.
    """

    def __init__(self, config: dict) -> None:
        self._enabled = config.get("enabled", True) and _MEDIAPIPE_AVAILABLE
        self._model_complexity = config.get("model_complexity", 0)
        self._min_detection_conf = config.get("min_detection_confidence", 0.5)
        self._min_tracking_conf = config.get("min_tracking_confidence", 0.5)
        self._wave_threshold = config.get("wave_threshold", 3)

        # MediaPipe Hands
        self._hands = None
        if self._enabled:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=4,
                model_complexity=self._model_complexity,
                min_detection_confidence=self._min_detection_conf,
                min_tracking_confidence=self._min_tracking_conf,
            )
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_hands = mp.solutions.hands
            logger.info("MediaPipe Hands 초기화 완료")

        # 손 흔들기 추적
        # track_id → WaveTracker
        self._wave_trackers: Dict[int, _WaveTracker] = {}

        # 손 흔드는 사람 목록
        self._waving_persons: List[WavingPerson] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_hands(self, frame: np.ndarray) -> List[HandInfo]:
        """
        프레임에서 손을 감지한다.

        Parameters
        ----------
        frame : np.ndarray
            BGR 이미지.

        Returns
        -------
        List[HandInfo]
            감지된 손 목록.
        """
        if not self._enabled or self._hands is None:
            return []

        h, w = frame.shape[:2]

        # RGB 변환
        rgb_frame = frame[..., ::-1]  # BGR → RGB

        # 손 감지
        results = self._hands.process(rgb_frame)

        hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # 랜드마크 추출
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                )

                # 손 중심
                center = (
                    float(np.mean(landmarks[:, 0])),
                    float(np.mean(landmarks[:, 1])),
                )

                # 바운딩 박스
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                x1 = int(max(0, np.min(x_coords) - 10))
                y1 = int(max(0, np.min(y_coords) - 10))
                x2 = int(min(w, np.max(x_coords) + 10))
                y2 = int(min(h, np.max(y_coords) + 10))

                hands.append(
                    HandInfo(
                        landmarks=landmarks,
                        handedness=handedness.classification[0].label,
                        center=center,
                        bbox=(x1, y1, x2, y2),
                        confidence=handedness.classification[0].score,
                    )
                )

        return hands

    def update_waving_detection(
        self,
        frame: np.ndarray,
        tracked_persons: List,
        camera_handler=None,
        depth_frame=None,
    ) -> List[WavingPerson]:
        """
        손 흔드는 사람을 감지하고 업데이트한다.

        Parameters
        ----------
        frame : np.ndarray
            BGR 이미지.
        tracked_persons : List[TrackedPerson]
            추적 중인 사람 목록.
        camera_handler : optional
            깊이 계산용 카메라 핸들러.
        depth_frame : optional
            깊이 프레임.

        Returns
        -------
        List[WavingPerson]
            손 흔드는 사람 목록 (가까운 순).
        """
        if not self._enabled:
            return []

        now = time.time()
        h, w = frame.shape[:2]

        # 1. 손 감지
        hands = self.detect_hands(frame)

        # 2. 각 손을 사람에 매칭
        for hand in hands:
            hand_center_px = (hand.center[0] * w, hand.center[1] * h)

            # 가장 가까운 사람 찾기
            best_person = None
            best_dist = float("inf")

            for person in tracked_persons:
                # 손이 사람 bbox 안에 있는지 확인
                px, py = int(hand_center_px[0]), int(hand_center_px[1])
                x1, y1, x2, y2 = person.bbox.astype(int)

                # bbox 확장 (팔 고려)
                x1 -= 50
                x2 += 50

                if x1 <= px <= x2 and y1 <= py <= y2:
                    dist = abs(person.center[0] - px) + abs(person.center[1] - py)
                    if dist < best_dist:
                        best_dist = dist
                        best_person = person

            if best_person is not None:
                # 손 흔들기 트래커 업데이트
                track_id = best_person.track_id
                if track_id not in self._wave_trackers:
                    self._wave_trackers[track_id] = _WaveTracker(track_id)

                self._wave_trackers[track_id].update(hand, now)

        # 3. 손 흔드는 사람 목록 생성
        self._waving_persons.clear()

        for person in tracked_persons:
            track_id = person.track_id
            if track_id in self._wave_trackers:
                tracker = self._wave_trackers[track_id]

                if tracker.is_waving(self._wave_threshold, now):
                    # 거리 계산
                    distance_m = None
                    if depth_frame is not None and camera_handler is not None:
                        x1, y1, x2, y2 = person.bbox.astype(int)
                        distance_m = camera_handler.get_median_depth_in_bbox(
                            depth_frame, x1, y1, x2, y2
                        )

                    self._waving_persons.append(
                        WavingPerson(
                            track_id=track_id,
                            center_x=person.center[0],
                            center_y=person.center[1],
                            distance_m=distance_m,
                            wave_count=tracker.wave_count,
                            last_wave_time=tracker.last_wave_time,
                            bbox=tuple(person.bbox.astype(int)),
                        )
                    )

        # 4. 거리순 정렬 (가까운 사람 우선)
        self._waving_persons.sort(
            key=lambda p: p.distance_m if p.distance_m else float("inf")
        )

        # 5. 오래된 트래커 정리
        self._cleanup_trackers(now)

        return self._waving_persons

    def get_nearest_waving_person(self) -> Optional[WavingPerson]:
        """가장 가까운 손 흔드는 사람을 반환한다."""
        if self._waving_persons:
            return self._waving_persons[0]
        return None

    def get_waving_person_dict(self) -> Optional[Dict]:
        """상태 머신용 딕셔너리 형태로 반환."""
        person = self.get_nearest_waving_person()
        if person is None:
            return None

        return {
            "track_id": person.track_id,
            "center_x": person.center_x,
            "distance_m": person.distance_m,
        }

    def draw_hands(self, frame: np.ndarray, hands: List[HandInfo]) -> np.ndarray:
        """프레임에 손 랜드마크를 그린다."""
        if not self._enabled or not hands:
            return frame

        vis = frame.copy()
        h, w = vis.shape[:2]

        for hand in hands:
            # 바운딩 박스
            x1, y1, x2, y2 = hand.bbox
            color = (0, 255, 0) if hand.handedness == "Right" else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 랜드마크 점
            for lm in hand.landmarks:
                px, py = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(vis, (px, py), 3, color, -1)

        return vis

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup_trackers(self, now: float, timeout: float = 5.0) -> None:
        """오래된 트래커를 정리한다."""
        expired = [
            tid
            for tid, tracker in self._wave_trackers.items()
            if now - tracker.last_update_time > timeout
        ]
        for tid in expired:
            del self._wave_trackers[tid]

    def stop(self) -> None:
        """리소스를 해제한다."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        logger.info("HandDetector 종료")


# ---------------------------------------------------------------------------
# Wave Tracker
# ---------------------------------------------------------------------------


class _WaveTracker:
    """손 흔들기 동작을 추적하는 내부 클래스."""

    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.wave_count = 0
        self.last_wave_time = 0.0
        self.last_update_time = 0.0

        # 손 y좌표 히스토리 (손 흔들기 = y좌표 진동)
        self._y_history: Deque[Tuple[float, float]] = deque(maxlen=30)
        self._last_direction: Optional[str] = None  # "up" | "down"
        self._direction_changes = 0

    def update(self, hand: HandInfo, now: float) -> None:
        """손 정보로 업데이트한다."""
        self.last_update_time = now

        # y좌표 기록
        y = hand.center[1]
        self._y_history.append((y, now))

        if len(self._y_history) < 5:
            return

        # 방향 변화 감지
        recent = list(self._y_history)[-5:]
        y_values = [item[0] for item in recent]

        # 현재 방향
        if y_values[-1] < y_values[-3]:
            current_direction = "up"
        elif y_values[-1] > y_values[-3]:
            current_direction = "down"
        else:
            current_direction = self._last_direction

        # 방향 변화 카운트
        if current_direction and current_direction != self._last_direction:
            self._direction_changes += 1
            self._last_direction = current_direction

            # 2번 방향 변화 = 1번 흔들기
            if self._direction_changes >= 2:
                self.wave_count += 1
                self.last_wave_time = now
                self._direction_changes = 0

    def is_waving(self, threshold: int, now: float) -> bool:
        """손 흔들기 여부를 반환한다."""
        # 최근 3초 내에 threshold 이상 흔들었으면 True
        if now - self.last_wave_time > 3.0:
            return False
        return self.wave_count >= threshold


# OpenCV import at module level for drawing
try:
    import cv2
except ImportError:
    cv2 = None
