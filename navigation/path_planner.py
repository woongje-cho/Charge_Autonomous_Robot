"""
Bridge Guardian — 경로 계획 모듈
순찰 웨이포인트 관리 및 대상 접근 경로 계획.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PatrolWaypoint:
    """
    순찰 경로 웨이포인트.

    Attributes
    ----------
    latitude : float
        위도 (WGS84).
    longitude : float
        경도 (WGS84).
    name : str
        지점 이름 (예: "마포대교 북단 1").
    pause_seconds : float
        해당 지점에서의 정지 시간 (초). 0이면 통과.
    scan_on_arrival : bool
        도착 시 주변 스캔 수행 여부.
    """

    latitude: float = 0.0
    longitude: float = 0.0
    name: str = ""
    pause_seconds: float = 0.0
    scan_on_arrival: bool = True


# ---------------------------------------------------------------------------
# PathPlanner
# ---------------------------------------------------------------------------

class PathPlanner:
    """
    순찰 경로 관리 및 대상 접근 경로 계획기.

    - 웨이포인트 목록 기반 순환 순찰
    - 위험 감지 시 대상 접근 경로 계산
    - GPS 비의존 폴백 (경로 인덱스 기반)

    Parameters
    ----------
    waypoints : list[PatrolWaypoint], optional
        초기 순찰 웨이포인트 목록.
    arrival_radius_m : float
        웨이포인트 도착 판정 반경 (미터).
    approach_speed : int
        접근 시 모터 속도 (0–255).
    patrol_speed : int
        순찰 시 모터 속도 (0–255).
    """

    def __init__(
        self,
        waypoints: Optional[List[PatrolWaypoint]] = None,
        arrival_radius_m: float = 3.0,
        approach_speed: int = 80,
        patrol_speed: int = 100,
    ) -> None:
        self._waypoints: List[PatrolWaypoint] = list(waypoints) if waypoints else []
        self._current_index: int = 0
        self._arrival_radius_m = arrival_radius_m
        self._approach_speed = approach_speed
        self._patrol_speed = patrol_speed
        self._paused_until: float = 0.0

        logger.info(
            "PathPlanner 초기화: %d개 웨이포인트, 도착 반경 %.1fm",
            len(self._waypoints),
            self._arrival_radius_m,
        )

    # ------------------------------------------------------------------
    # Waypoint management
    # ------------------------------------------------------------------

    def add_waypoint(self, waypoint: PatrolWaypoint) -> None:
        """순찰 웨이포인트를 추가한다."""
        self._waypoints.append(waypoint)
        logger.info("웨이포인트 추가: %s (총 %d개)", waypoint.name, len(self._waypoints))

    def clear_waypoints(self) -> None:
        """모든 웨이포인트를 제거한다."""
        self._waypoints.clear()
        self._current_index = 0
        logger.info("모든 웨이포인트 제거됨")

    def set_waypoints(self, waypoints: List[PatrolWaypoint]) -> None:
        """웨이포인트 목록을 교체한다."""
        self._waypoints = list(waypoints)
        self._current_index = 0
        logger.info("웨이포인트 목록 설정: %d개", len(self._waypoints))

    @property
    def waypoint_count(self) -> int:
        """등록된 웨이포인트 수"""
        return len(self._waypoints)

    @property
    def current_waypoint(self) -> Optional[PatrolWaypoint]:
        """현재 목표 웨이포인트"""
        if not self._waypoints:
            return None
        return self._waypoints[self._current_index % len(self._waypoints)]

    @property
    def current_index(self) -> int:
        """현재 웨이포인트 인덱스"""
        return self._current_index

    # ------------------------------------------------------------------
    # Patrol navigation
    # ------------------------------------------------------------------

    def update_patrol(
        self,
        current_lat: float,
        current_lon: float,
    ) -> dict:
        """
        순찰 상태를 갱신하고 다음 동작을 반환한다.

        Parameters
        ----------
        current_lat : float
            현재 위도.
        current_lon : float
            현재 경도.

        Returns
        -------
        dict
            ``{"arrived": bool, "steering": int, "speed": int,
              "waypoint": PatrolWaypoint|None, "distance_m": float,
              "bearing_deg": float, "paused": bool}``
        """
        result = {
            "arrived": False,
            "steering": 0,
            "speed": 0,
            "waypoint": None,
            "distance_m": 0.0,
            "bearing_deg": 0.0,
            "paused": False,
        }

        target = self.current_waypoint
        if target is None:
            return result

        result["waypoint"] = target

        # 정지 상태 확인
        now = time.time()
        if now < self._paused_until:
            result["paused"] = True
            return result

        # 거리 / 방위 계산
        dist = self._haversine(current_lat, current_lon, target.latitude, target.longitude)
        bearing = self._bearing(current_lat, current_lon, target.latitude, target.longitude)
        result["distance_m"] = dist
        result["bearing_deg"] = bearing

        # 도착 판정
        if dist <= self._arrival_radius_m:
            result["arrived"] = True
            logger.info(
                "웨이포인트 도착: [%d] %s (%.1fm)",
                self._current_index, target.name, dist,
            )

            # 정지 시간
            if target.pause_seconds > 0:
                self._paused_until = now + target.pause_seconds
                result["paused"] = True

            # 다음 웨이포인트로 순환
            self._current_index = (self._current_index + 1) % len(self._waypoints)
            return result

        # 조향 계산 (간이: 방위각 기반)
        steering = self._bearing_to_steering(bearing)
        result["steering"] = steering
        result["speed"] = self._patrol_speed
        return result

    # ------------------------------------------------------------------
    # Approach planning (대상 접근)
    # ------------------------------------------------------------------

    def plan_approach(
        self,
        current_lat: float,
        current_lon: float,
        target_lat: float,
        target_lon: float,
    ) -> dict:
        """
        감지된 대상에게 접근하기 위한 경로를 계산한다.

        Parameters
        ----------
        current_lat, current_lon : float
            로봇 현재 위치.
        target_lat, target_lon : float
            대상 위치.

        Returns
        -------
        dict
            ``{"steering": int, "speed": int, "distance_m": float,
              "bearing_deg": float, "reached": bool}``
        """
        dist = self._haversine(current_lat, current_lon, target_lat, target_lon)
        bearing = self._bearing(current_lat, current_lon, target_lat, target_lon)
        steering = self._bearing_to_steering(bearing)

        reached = dist <= self._arrival_radius_m

        # 가까울수록 느리게 접근
        if dist < self._arrival_radius_m * 2:
            speed = max(40, self._approach_speed // 2)
        else:
            speed = self._approach_speed

        if reached:
            speed = 0
            steering = 0

        return {
            "steering": steering,
            "speed": speed,
            "distance_m": dist,
            "bearing_deg": bearing,
            "reached": reached,
        }

    def plan_approach_pixel(
        self,
        bbox_center_x: float,
        frame_width: int,
    ) -> dict:
        """
        카메라 프레임 기반 간이 접근 (GPS 없이 사용 가능).

        대상의 바운딩 박스 중심 x좌표와 프레임 폭을 사용하여
        조향 방향을 결정한다.

        Parameters
        ----------
        bbox_center_x : float
            대상 바운딩 박스 중심의 x 좌표 (픽셀).
        frame_width : int
            프레임 너비 (픽셀).

        Returns
        -------
        dict
            ``{"steering": int, "speed": int}``
        """
        if frame_width <= 0:
            return {"steering": 0, "speed": 0}

        # 정규화: -1.0(왼쪽) ~ +1.0(오른쪽)
        normalized = (bbox_center_x - frame_width / 2) / (frame_width / 2)
        # 조향값 매핑 (-7 ~ +7)
        steering = int(round(normalized * 7))
        steering = max(-7, min(7, steering))

        return {
            "steering": steering,
            "speed": self._approach_speed,
        }

    # ------------------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 GPS 좌표 간 거리 (미터) — Haversine 공식."""
        R = 6_371_000  # 지구 반지름 (m)
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 GPS 좌표 간 방위각 (도, 0=북, 시계방향)."""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_lambda = math.radians(lon2 - lon1)

        x = math.sin(d_lambda) * math.cos(phi2)
        y = (
            math.cos(phi1) * math.sin(phi2)
            - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
        )
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @staticmethod
    def _bearing_to_steering(bearing_deg: float) -> int:
        """
        방위각(도)을 조향값 (-7~+7)으로 변환한다.

        0° / 360° = 직진, 양수 = 우회전, 음수 = 좌회전.
        """
        # -180 ~ +180 범위로 정규화
        angle = bearing_deg
        if angle > 180:
            angle -= 360

        # ±180° 범위에서 ±7 매핑
        steering = int(round(angle / 180 * 7))
        return max(-7, min(7, steering))
