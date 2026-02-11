"""
Bridge Guardian — 모터 제어 모듈
H-Mobility 프로토콜 기반 시리얼 통신 (s{steering}l{left}r{right}\\n).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (H-Mobility 프로토콜 범위)
# ---------------------------------------------------------------------------
STEERING_MIN = -7
STEERING_MAX = 7
SPEED_MIN = 0
SPEED_MAX = 255
DEFAULT_BAUDRATE = 9600


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MotorCommand:
    """
    모터 명령 데이터.

    Attributes
    ----------
    steering : int
        조향값 (-7 = 최대 좌회전, 0 = 직진, +7 = 최대 우회전).
    left_speed : int
        좌측 모터 속도 (0–255).
    right_speed : int
        우측 모터 속도 (0–255).
    """

    steering: int = 0
    left_speed: int = 0
    right_speed: int = 0

    def __post_init__(self) -> None:
        self.steering = max(STEERING_MIN, min(STEERING_MAX, int(self.steering)))
        self.left_speed = max(SPEED_MIN, min(SPEED_MAX, int(self.left_speed)))
        self.right_speed = max(SPEED_MIN, min(SPEED_MAX, int(self.right_speed)))

    def to_serial(self) -> str:
        """
        H-Mobility 시리얼 프로토콜 메시지로 변환한다.

        Returns
        -------
        str
            ``"s{steering}l{left_speed}r{right_speed}\\n"``
        """
        return f"s{self.steering}l{self.left_speed}r{self.right_speed}\n"

    @property
    def is_stop(self) -> bool:
        """정지 명령 여부"""
        return self.left_speed == 0 and self.right_speed == 0


# ---------------------------------------------------------------------------
# MotorController
# ---------------------------------------------------------------------------

class MotorController:
    """
    시리얼 포트를 통한 모터 제어기.

    H-Mobility 프로토콜: ``s{steering}l{left_speed}r{right_speed}\\n``
    9600 baud, 아두이노 호환.

    Parameters
    ----------
    port : str
        시리얼 포트 경로.
    baudrate : int
        보드레이트 (기본 9600).
    """

    # 비상 정지 명령 (상수)
    STOP_COMMAND = MotorCommand(steering=0, left_speed=0, right_speed=0)

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = DEFAULT_BAUDRATE,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._lock = threading.Lock()
        self._serial = None
        self._connected = False
        self._last_command: Optional[MotorCommand] = None

        # pyserial 선택 임포트
        self._serial_mod = self._try_import("serial")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_import(module_name: str):
        """모듈을 선택적으로 임포트한다."""
        try:
            import importlib
            return importlib.import_module(module_name)
        except ImportError:
            logger.warning(
                "%s 모듈을 찾을 수 없습니다. 모터 명령이 로그로만 출력됩니다.",
                module_name,
            )
            return None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """시리얼 포트를 열고 연결한다."""
        if self._serial_mod is None:
            logger.warning("pyserial 없음 — 시뮬레이션 모드로 동작합니다.")
            self._connected = False
            return False

        try:
            self._serial = self._serial_mod.Serial(
                self._port, self._baudrate, timeout=1,
            )
            time.sleep(1.0)  # 아두이노 부트 대기 (H-Mobility 패턴)
            self._connected = True
            logger.info(
                "모터 시리얼 포트 연결됨: %s @ %d baud", self._port, self._baudrate,
            )
            return True
        except Exception:
            logger.exception("모터 시리얼 포트 열기 실패: %s", self._port)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """정지 명령을 보낸 뒤 시리얼 포트를 닫는다."""
        self.send(self.STOP_COMMAND)
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._connected = False
        logger.info("모터 시리얼 포트 닫힘")

    @property
    def is_connected(self) -> bool:
        """시리얼 연결 상태"""
        return self._connected and self._serial is not None

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def send(self, command: MotorCommand) -> bool:
        """
        모터 명령을 시리얼로 전송한다.

        Parameters
        ----------
        command : MotorCommand
            전송할 명령.

        Returns
        -------
        bool
            전송 성공 여부. 시리얼 미연결 시 로그만 출력하고 False.
        """
        msg = command.to_serial()
        self._last_command = command

        with self._lock:
            if self._serial is not None and self._connected:
                try:
                    self._serial.write(msg.encode())
                    logger.debug("모터 명령 전송: %s", msg.strip())
                    return True
                except Exception:
                    logger.exception("모터 명령 전송 실패")
                    self._connected = False
                    return False
            else:
                logger.debug(
                    "모터 시뮬레이션: %s", msg.strip(),
                )
                return False

    def stop(self) -> bool:
        """비상 정지 명령을 전송한다."""
        logger.info("비상 정지 명령 전송")
        return self.send(self.STOP_COMMAND)

    def drive(self, steering: int, speed: int) -> bool:
        """
        좌/우 동일 속도로 주행한다.

        Parameters
        ----------
        steering : int
            조향값 (-7 ~ +7).
        speed : int
            좌우 공통 속도 (0–255).
        """
        cmd = MotorCommand(
            steering=steering,
            left_speed=speed,
            right_speed=speed,
        )
        return self.send(cmd)

    @property
    def last_command(self) -> Optional[MotorCommand]:
        """마지막으로 전송한 명령"""
        return self._last_command
