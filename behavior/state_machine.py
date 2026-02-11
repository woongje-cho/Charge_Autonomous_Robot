"""
ChargeMate - 충전 로봇 행동 상태 머신
로봇의 전체 행동 상태를 관리하고, 센서/요청 입력에 따라 행동을 결정한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from navigation.motor_controller import MotorCommand

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Robot State Enum
# ---------------------------------------------------------------------------


class RobotState(Enum):
    """충전 로봇 행동 상태"""

    IDLE = auto()  # 대기 (홈에서 대기 중)
    DISPATCHING = auto()  # 배차 (요청 수락, 출발 준비)
    NAVIGATING = auto()  # 이동 중 (사용자 위치로 이동)
    APPROACHING_USER = auto()  # 사용자 접근 (손 흔드는 사람 찾아 접근)
    DOCKING = auto()  # 도킹 (충전 케이블 연결 대기)
    CHARGING = auto()  # 충전 중
    UNDOCKING = auto()  # 언도킹 (충전 완료, 분리 대기)
    RETURNING = auto()  # 복귀 중 (홈으로 이동)
    EMERGENCY_STOP = auto()  # 비상 정지


# ---------------------------------------------------------------------------
# TTS Messages
# ---------------------------------------------------------------------------

_TTS_MESSAGES: Dict[str, List[str]] = {
    "dispatching": [
        "충전 요청을 받았습니다. 곧 출발합니다.",
    ],
    "arriving": [
        "목적지 근처에 도착했습니다. 손을 흔들어 주세요.",
    ],
    "approaching": [
        "사용자를 발견했습니다. 다가가고 있습니다.",
    ],
    "docking": [
        "충전 케이블을 연결해 주세요.",
        "USB-C 케이블을 기기에 꽂아주세요.",
    ],
    "charging": [
        "충전을 시작합니다.",
    ],
    "charging_progress": [
        "현재 충전량 {soc}% 입니다.",
    ],
    "charging_complete": [
        "충전이 완료되었습니다. 케이블을 분리해 주세요.",
    ],
    "undocking": [
        "케이블이 분리되었습니다. 감사합니다.",
    ],
    "returning": [
        "홈 위치로 복귀합니다.",
    ],
    "low_battery": [
        "로봇 배터리가 부족합니다. 충전소로 복귀합니다.",
    ],
    "emergency": [
        "비상 정지가 활성화되었습니다.",
    ],
}


# ---------------------------------------------------------------------------
# State Context
# ---------------------------------------------------------------------------


@dataclass
class _StateContext:
    """상태 머신 내부 컨텍스트"""

    state: RobotState = RobotState.IDLE
    previous_state: RobotState = RobotState.IDLE
    state_enter_time: float = 0.0

    # 현재 요청 정보
    current_request_id: Optional[str] = None
    target_latitude: float = 0.0
    target_longitude: float = 0.0
    user_name: Optional[str] = None

    # 감지된 사용자 정보
    detected_user_track_id: Optional[int] = None
    detected_user_center_x: float = 0.0
    detected_user_distance: float = 0.0

    # 충전 상태
    charging_start_time: float = 0.0
    last_soc_announce_time: float = 0.0
    current_soc: float = 0.0

    # TTS 상태
    last_tts_time: float = 0.0
    tts_message_index: int = 0


# ---------------------------------------------------------------------------
# Behavior State Machine
# ---------------------------------------------------------------------------


class ChargingStateMachine:
    """
    ChargeMate 충전 로봇 행동 상태 머신.

    ``update()``를 매 프레임 호출하면, 현재 센서 입력에 따라
    상태를 전이하고 실행할 행동(action dict)을 반환한다.

    Parameters
    ----------
    config : dict
        behavior 설정 딕셔너리.
    """

    def __init__(self, config: dict = None) -> None:
        cfg = config or {}
        self._ctx = _StateContext()

        # 타임아웃 설정
        self._dispatch_timeout = cfg.get("dispatch_timeout_s", 300)
        self._navigation_timeout = cfg.get("navigation_timeout_s", 600)
        self._approaching_timeout = cfg.get("approaching_timeout_s", 120)
        self._docking_timeout = cfg.get("docking_timeout_s", 60)
        self._charging_timeout = cfg.get("charging_timeout_s", 7200)

        # TTS 간격
        self._tts_interval = cfg.get("tts_interval_s", 10.0)
        self._soc_announce_interval = cfg.get("soc_announce_interval_s", 300.0)  # 5분

        # 홈 위치
        self._home_lat = cfg.get("home_latitude", 37.5665)
        self._home_lon = cfg.get("home_longitude", 126.9780)

        logger.info(
            "ChargingStateMachine 초기화: dispatch_timeout=%.0fs, charging_timeout=%.0fs",
            self._dispatch_timeout,
            self._charging_timeout,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> RobotState:
        """현재 로봇 상태"""
        return self._ctx.state

    @property
    def previous_state(self) -> RobotState:
        """이전 로봇 상태"""
        return self._ctx.previous_state

    @property
    def state_duration(self) -> float:
        """현재 상태 유지 시간 (초)"""
        return time.time() - self._ctx.state_enter_time

    @property
    def current_request_id(self) -> Optional[str]:
        """현재 처리 중인 요청 ID"""
        return self._ctx.current_request_id

    @property
    def home_position(self) -> Tuple[float, float]:
        """홈 위치 (latitude, longitude)"""
        return (self._home_lat, self._home_lon)

    # ------------------------------------------------------------------
    # State Transitions
    # ------------------------------------------------------------------

    def _transition(self, new_state: RobotState) -> None:
        """상태를 전이한다."""
        if new_state == self._ctx.state:
            return

        old = self._ctx.state
        self._ctx.previous_state = old
        self._ctx.state = new_state
        self._ctx.state_enter_time = time.time()
        self._ctx.tts_message_index = 0
        self._ctx.last_tts_time = 0.0

        logger.info("상태 전이: %s → %s", old.name, new_state.name)

        # 상태 진입 시 초기화
        if new_state == RobotState.CHARGING:
            self._ctx.charging_start_time = time.time()
            self._ctx.last_soc_announce_time = time.time()

    def force_state(self, state: RobotState) -> None:
        """상태를 강제 전이한다 (외부 제어용)."""
        logger.warning("강제 상태 전이: %s → %s", self._ctx.state.name, state.name)
        self._transition(state)

    def emergency_stop(self) -> dict:
        """비상 정지 상태로 전이하고 정지 명령을 반환한다."""
        self._transition(RobotState.EMERGENCY_STOP)
        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message="비상 정지가 활성화되었습니다.",
        )

    # ------------------------------------------------------------------
    # Request Handling
    # ------------------------------------------------------------------

    def accept_request(
        self,
        request_id: str,
        latitude: float,
        longitude: float,
        user_name: Optional[str] = None,
    ) -> bool:
        """충전 요청을 수락한다."""
        if self._ctx.state != RobotState.IDLE:
            logger.warning("IDLE 상태가 아니어서 요청 거부: %s", request_id)
            return False

        self._ctx.current_request_id = request_id
        self._ctx.target_latitude = latitude
        self._ctx.target_longitude = longitude
        self._ctx.user_name = user_name

        self._transition(RobotState.DISPATCHING)
        logger.info(
            "요청 수락: %s (%.6f, %.6f) - %s",
            request_id,
            latitude,
            longitude,
            user_name or "익명",
        )
        return True

    def cancel_request(self) -> None:
        """현재 요청을 취소하고 홈으로 복귀한다."""
        if self._ctx.current_request_id:
            logger.info("요청 취소: %s", self._ctx.current_request_id)
        self._ctx.current_request_id = None
        self._ctx.target_latitude = 0.0
        self._ctx.target_longitude = 0.0
        self._ctx.detected_user_track_id = None
        self._transition(RobotState.RETURNING)

    # ------------------------------------------------------------------
    # Main Update
    # ------------------------------------------------------------------

    def update(
        self,
        current_lat: float = 0.0,
        current_lon: float = 0.0,
        distance_to_target: float = float("inf"),
        distance_to_home: float = float("inf"),
        detected_waving_user: Optional[Dict] = None,
        is_charging: bool = False,
        charging_current_ma: float = 0.0,
        output_soc: float = 0.0,
        robot_battery_low: bool = False,
        navigation_steering: int = 0,
        navigation_speed: int = 80,
    ) -> dict:
        """
        매 프레임 호출되어 현재 입력에 따라 상태를 전이하고 행동을 결정한다.

        Parameters
        ----------
        current_lat, current_lon : float
            현재 GPS 위치.
        distance_to_target : float
            목표 지점까지 거리 (미터).
        distance_to_home : float
            홈까지 거리 (미터).
        detected_waving_user : dict, optional
            손 흔드는 사용자 정보 {"track_id", "center_x", "distance_m"}.
        is_charging : bool
            충전 중 여부.
        charging_current_ma : float
            현재 충전 전류 (mA).
        output_soc : float
            출력 배터리 SoC (%).
        robot_battery_low : bool
            로봇 배터리 부족 여부.
        navigation_steering : int
            네비게이션 조향값 (-7 ~ +7).
        navigation_speed : int
            네비게이션 속도 (0-255).

        Returns
        -------
        dict
            행동 딕셔너리.
        """
        now = time.time()
        state = self._ctx.state

        # ---- 비상 정지: 외부에서만 해제 가능 ----
        if state == RobotState.EMERGENCY_STOP:
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            )

        # ---- 로봇 배터리 부족 시 강제 복귀 ----
        if robot_battery_low and state not in (
            RobotState.RETURNING,
            RobotState.IDLE,
            RobotState.EMERGENCY_STOP,
        ):
            logger.warning("로봇 배터리 부족 - 강제 복귀")
            self.cancel_request()
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                tts_message=_TTS_MESSAGES["low_battery"][0],
            )

        # ---- 상태별 처리 ----
        if state == RobotState.IDLE:
            return self._handle_idle()

        elif state == RobotState.DISPATCHING:
            return self._handle_dispatching(now)

        elif state == RobotState.NAVIGATING:
            return self._handle_navigating(
                now, distance_to_target, navigation_steering, navigation_speed
            )

        elif state == RobotState.APPROACHING_USER:
            return self._handle_approaching_user(now, detected_waving_user)

        elif state == RobotState.DOCKING:
            return self._handle_docking(now, is_charging)

        elif state == RobotState.CHARGING:
            return self._handle_charging(
                now, is_charging, charging_current_ma, output_soc
            )

        elif state == RobotState.UNDOCKING:
            return self._handle_undocking(now, is_charging)

        elif state == RobotState.RETURNING:
            return self._handle_returning(
                now, distance_to_home, navigation_steering, navigation_speed
            )

        # fallback
        return self._make_action()

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _handle_idle(self) -> dict:
        """IDLE: 홈에서 대기."""
        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
        )

    def _handle_dispatching(self, now: float) -> dict:
        """DISPATCHING: 출발 준비."""
        # 타임아웃 체크
        if self.state_duration > self._dispatch_timeout:
            logger.warning("배차 타임아웃 - IDLE로 복귀")
            self._ctx.current_request_id = None
            self._transition(RobotState.IDLE)
            return self._make_action()

        # TTS 안내
        tts_msg = None
        if now - self._ctx.last_tts_time >= self._tts_interval:
            tts_msg = _TTS_MESSAGES["dispatching"][0]
            self._ctx.last_tts_time = now

        # 바로 NAVIGATING으로 전이
        self._transition(RobotState.NAVIGATING)

        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message=tts_msg,
        )

    def _handle_navigating(
        self,
        now: float,
        distance_to_target: float,
        steering: int,
        speed: int,
    ) -> dict:
        """NAVIGATING: 목표 지점으로 이동."""
        # 타임아웃 체크
        if self.state_duration > self._navigation_timeout:
            logger.warning("네비게이션 타임아웃 - 복귀")
            self.cancel_request()
            return self._make_action(
                tts_message="목적지에 도달하지 못했습니다. 복귀합니다.",
            )

        # 도착 체크 (3m 이내)
        if distance_to_target < 3.0:
            logger.info(
                "목적지 근처 도착 (%.1fm) - 사용자 탐색 시작", distance_to_target
            )
            self._transition(RobotState.APPROACHING_USER)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                tts_message=_TTS_MESSAGES["arriving"][0],
            )

        return self._make_action(
            motor_command=MotorCommand(
                steering=steering,
                left_speed=speed,
                right_speed=speed,
            ),
        )

    def _handle_approaching_user(
        self,
        now: float,
        detected_user: Optional[Dict],
    ) -> dict:
        """APPROACHING_USER: 손 흔드는 사용자에게 접근."""
        # 타임아웃 체크
        if self.state_duration > self._approaching_timeout:
            logger.warning("사용자 접근 타임아웃 - 복귀")
            self.cancel_request()
            return self._make_action(
                tts_message="사용자를 찾지 못했습니다. 복귀합니다.",
            )

        # 사용자 감지됨
        if detected_user is not None:
            self._ctx.detected_user_track_id = detected_user.get("track_id")
            self._ctx.detected_user_center_x = detected_user.get("center_x", 0)
            self._ctx.detected_user_distance = detected_user.get("distance_m", 0)

            # 가까이 접근 완료
            if self._ctx.detected_user_distance < 1.5:
                logger.info(
                    "사용자 접근 완료 (%.1fm)", self._ctx.detected_user_distance
                )
                self._transition(RobotState.DOCKING)
                return self._make_action(
                    motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                    tts_message=_TTS_MESSAGES["docking"][0],
                )

            # 사용자 방향으로 이동
            # center_x 기반 조향 (-1.0 ~ +1.0 → -7 ~ +7)
            normalized_x = (self._ctx.detected_user_center_x - 320) / 320  # 640px 기준
            steering = int(normalized_x * 7)
            steering = max(-7, min(7, steering))

            # TTS 안내
            tts_msg = None
            if self._ctx.tts_message_index == 0:
                tts_msg = _TTS_MESSAGES["approaching"][0]
                self._ctx.tts_message_index = 1
                self._ctx.last_tts_time = now

            return self._make_action(
                motor_command=MotorCommand(
                    steering=steering,
                    left_speed=50,
                    right_speed=50,
                ),
                tts_message=tts_msg,
            )

        # 사용자 미감지 - 제자리 회전하며 탐색
        return self._make_action(
            motor_command=MotorCommand(steering=3, left_speed=30, right_speed=30),
        )

    def _handle_docking(self, now: float, is_charging: bool) -> dict:
        """DOCKING: 충전 케이블 연결 대기."""
        # 타임아웃 체크
        if self.state_duration > self._docking_timeout:
            logger.warning("도킹 타임아웃 - 복귀")
            self.cancel_request()
            return self._make_action(
                tts_message="케이블이 연결되지 않았습니다. 복귀합니다.",
            )

        # 충전 감지
        if is_charging:
            logger.info("충전 시작 감지")
            self._transition(RobotState.CHARGING)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                tts_message=_TTS_MESSAGES["charging"][0],
            )

        # TTS 반복 안내
        tts_msg = None
        if now - self._ctx.last_tts_time >= 15.0:  # 15초마다
            messages = _TTS_MESSAGES["docking"]
            self._ctx.tts_message_index = (self._ctx.tts_message_index + 1) % len(
                messages
            )
            tts_msg = messages[self._ctx.tts_message_index]
            self._ctx.last_tts_time = now

        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message=tts_msg,
        )

    def _handle_charging(
        self,
        now: float,
        is_charging: bool,
        current_ma: float,
        soc: float,
    ) -> dict:
        """CHARGING: 충전 중."""
        self._ctx.current_soc = soc

        # 충전 타임아웃
        if self.state_duration > self._charging_timeout:
            logger.warning("충전 타임아웃 (%.0f초) - 강제 종료", self.state_duration)
            self._transition(RobotState.UNDOCKING)
            return self._make_action(
                tts_message="충전 시간이 초과되었습니다. 케이블을 분리해 주세요.",
            )

        # 충전 완료 (전류 50mA 미만 5초 지속)
        if current_ma < 50 and is_charging:
            logger.info("충전 완료 감지 (전류: %.1fmA)", current_ma)
            self._transition(RobotState.UNDOCKING)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                tts_message=_TTS_MESSAGES["charging_complete"][0],
            )

        # 케이블 분리됨 (충전 중단)
        if not is_charging:
            logger.info("충전 중단 감지 (케이블 분리)")
            self._transition(RobotState.UNDOCKING)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                tts_message="충전이 중단되었습니다.",
            )

        # SoC 주기적 안내
        tts_msg = None
        if now - self._ctx.last_soc_announce_time >= self._soc_announce_interval:
            tts_msg = f"현재 충전량 {int(soc)}% 입니다."
            self._ctx.last_soc_announce_time = now

        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message=tts_msg,
            charging_soc=soc,
        )

    def _handle_undocking(self, now: float, is_charging: bool) -> dict:
        """UNDOCKING: 케이블 분리 대기."""
        # 케이블 분리 확인
        if not is_charging:
            logger.info("케이블 분리 확인 - 복귀 시작")
            self._ctx.current_request_id = None
            self._transition(RobotState.RETURNING)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                tts_message=_TTS_MESSAGES["undocking"][0],
            )

        # TTS 반복 안내
        tts_msg = None
        if now - self._ctx.last_tts_time >= 10.0:
            tts_msg = _TTS_MESSAGES["charging_complete"][0]
            self._ctx.last_tts_time = now

        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message=tts_msg,
        )

    def _handle_returning(
        self,
        now: float,
        distance_to_home: float,
        steering: int,
        speed: int,
    ) -> dict:
        """RETURNING: 홈으로 복귀."""
        # 홈 도착 체크
        if distance_to_home < 2.0:
            logger.info("홈 도착 (%.1fm)", distance_to_home)
            self._transition(RobotState.IDLE)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                tts_message="홈 위치에 도착했습니다. 대기 모드로 전환합니다.",
            )

        # TTS 안내
        tts_msg = None
        if self._ctx.tts_message_index == 0:
            tts_msg = _TTS_MESSAGES["returning"][0]
            self._ctx.tts_message_index = 1

        return self._make_action(
            motor_command=MotorCommand(
                steering=steering,
                left_speed=speed,
                right_speed=speed,
            ),
            tts_message=tts_msg,
        )

    # ------------------------------------------------------------------
    # Action Builder
    # ------------------------------------------------------------------

    @staticmethod
    def _make_action(
        motor_command: Optional[MotorCommand] = None,
        tts_message: Optional[str] = None,
        charging_soc: Optional[float] = None,
    ) -> dict:
        """행동 딕셔너리를 생성한다."""
        if motor_command is None:
            motor_command = MotorCommand(steering=0, left_speed=0, right_speed=0)

        return {
            "motor_command": motor_command,
            "tts_message": tts_message,
            "charging_soc": charging_soc,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """현재 상태 요약 딕셔너리를 반환한다."""
        return {
            "state": self._ctx.state.name,
            "previous_state": self._ctx.previous_state.name,
            "state_duration_s": round(self.state_duration, 1),
            "current_request_id": self._ctx.current_request_id,
            "target_latitude": self._ctx.target_latitude,
            "target_longitude": self._ctx.target_longitude,
            "user_name": self._ctx.user_name,
            "detected_user_track_id": self._ctx.detected_user_track_id,
            "current_soc": self._ctx.current_soc,
        }

    def reset(self) -> None:
        """상태 머신을 IDLE로 초기화한다."""
        self._ctx = _StateContext()
        logger.info("상태 머신 초기화됨 (IDLE)")
