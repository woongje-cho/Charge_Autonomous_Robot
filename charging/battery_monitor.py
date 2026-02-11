"""
ChargeMate - 배터리 모니터링 모듈
로봇 자체 배터리 및 충전 출력 배터리 상태를 모니터링한다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graceful Imports
# ---------------------------------------------------------------------------

_INA219_AVAILABLE = False

try:
    # Raspberry Pi / Linux I2C
    from ina219 import INA219

    _INA219_AVAILABLE = True
except ImportError:
    pass

if not _INA219_AVAILABLE:
    try:
        # Alternative: smbus2 direct access
        import smbus2

        _SMBUS_AVAILABLE = True
    except ImportError:
        _SMBUS_AVAILABLE = False
        logger.warning("INA219/smbus2 사용 불가 - Mock 모드로 동작")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


class BatteryType(Enum):
    """배터리 유형"""

    ROBOT = "robot"  # 로봇 구동 배터리
    OUTPUT = "output"  # 충전 출력 배터리


@dataclass
class BatteryStatus:
    """배터리 상태 데이터"""

    battery_type: BatteryType
    voltage: float = 0.0  # 전압 (V)
    current_ma: float = 0.0  # 전류 (mA), 양수=방전, 음수=충전
    power_mw: float = 0.0  # 전력 (mW)
    soc: float = 0.0  # 충전 상태 (%)
    is_charging: bool = False  # 충전 중 여부
    is_low: bool = False  # 저전력 여부
    is_critical: bool = False  # 긴급 여부
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Battery Monitor
# ---------------------------------------------------------------------------


class BatteryMonitor:
    """
    배터리 모니터링 클래스.

    INA219 센서 또는 Mock 모드로 동작한다.

    Parameters
    ----------
    config : dict
        배터리 설정 딕셔너리.
    battery_type : BatteryType
        모니터링할 배터리 유형.
    """

    def __init__(
        self,
        config: dict,
        battery_type: BatteryType = BatteryType.ROBOT,
    ) -> None:
        self._config = config
        self._battery_type = battery_type
        self._sensor_type = config.get("sensor_type", "mock")
        self._i2c_address = config.get("i2c_address", 0x40)

        # 전압 범위
        self._min_voltage = config.get("min_voltage", 10.5)
        self._max_voltage = config.get("max_voltage", 12.6)

        # 임계값
        self._low_threshold = config.get("low_battery_threshold", 20)
        self._critical_threshold = config.get("critical_threshold", 10)

        # 용량 (출력 배터리용)
        self._capacity_mah = config.get("capacity_mah", 20000)

        # 센서 초기화
        self._sensor = None
        self._init_sensor()

        # 상태
        self._lock = threading.Lock()
        self._latest_status = BatteryStatus(battery_type=battery_type)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Mock 시뮬레이션용
        self._mock_voltage = self._max_voltage * 0.8
        self._mock_current = 0.0
        self._mock_soc = 80.0

        logger.info(
            "BatteryMonitor 초기화: type=%s, sensor=%s, range=[%.1f, %.1f]V",
            battery_type.value,
            self._sensor_type,
            self._min_voltage,
            self._max_voltage,
        )

    # ------------------------------------------------------------------
    # Sensor Initialization
    # ------------------------------------------------------------------

    def _init_sensor(self) -> None:
        """센서를 초기화한다."""
        if self._sensor_type == "ina219" and _INA219_AVAILABLE:
            try:
                self._sensor = INA219(
                    shunt_ohms=0.1,
                    max_expected_amps=3.0,
                    address=self._i2c_address,
                )
                self._sensor.configure()
                logger.info("INA219 센서 초기화 완료 (주소: 0x%02X)", self._i2c_address)
            except Exception as exc:
                logger.warning("INA219 초기화 실패: %s - Mock 모드로 전환", exc)
                self._sensor_type = "mock"
        else:
            self._sensor_type = "mock"
            logger.info("배터리 모니터링 Mock 모드")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """백그라운드 모니터링 스레드를 시작한다."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name=f"battery-{self._battery_type.value}",
            daemon=True,
        )
        self._thread.start()
        logger.info("배터리 모니터링 시작: %s", self._battery_type.value)

    def stop(self) -> None:
        """모니터링을 중지한다."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("배터리 모니터링 중지: %s", self._battery_type.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self) -> BatteryStatus:
        """최신 배터리 상태를 반환한다."""
        with self._lock:
            return BatteryStatus(
                battery_type=self._latest_status.battery_type,
                voltage=self._latest_status.voltage,
                current_ma=self._latest_status.current_ma,
                power_mw=self._latest_status.power_mw,
                soc=self._latest_status.soc,
                is_charging=self._latest_status.is_charging,
                is_low=self._latest_status.is_low,
                is_critical=self._latest_status.is_critical,
                timestamp=self._latest_status.timestamp,
            )

    def is_low_battery(self) -> bool:
        """저전력 여부를 반환한다."""
        return self._latest_status.is_low

    def is_critical(self) -> bool:
        """긴급 여부를 반환한다."""
        return self._latest_status.is_critical

    def is_charging(self) -> bool:
        """충전 중 여부를 반환한다."""
        return self._latest_status.is_charging

    def get_soc(self) -> float:
        """현재 충전 상태(%)를 반환한다."""
        return self._latest_status.soc

    def get_current_ma(self) -> float:
        """현재 전류(mA)를 반환한다."""
        return self._latest_status.current_ma

    # ------------------------------------------------------------------
    # Mock Simulation
    # ------------------------------------------------------------------

    def simulate_charging(self, is_charging: bool, current_ma: float = 500) -> None:
        """Mock 모드에서 충전 상태를 시뮬레이션한다."""
        if self._sensor_type != "mock":
            return

        with self._lock:
            if is_charging:
                self._mock_current = -abs(current_ma)  # 음수 = 충전
                self._mock_soc = min(100, self._mock_soc + 0.1)
            else:
                self._mock_current = 0

    def simulate_discharge(self, current_ma: float = 100) -> None:
        """Mock 모드에서 방전을 시뮬레이션한다."""
        if self._sensor_type != "mock":
            return

        with self._lock:
            self._mock_current = abs(current_ma)  # 양수 = 방전
            self._mock_soc = max(0, self._mock_soc - 0.01)

    # ------------------------------------------------------------------
    # Monitoring Loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """배터리 상태를 주기적으로 읽는다."""
        while self._running:
            try:
                if self._sensor_type == "ina219" and self._sensor is not None:
                    self._read_ina219()
                else:
                    self._read_mock()
            except Exception as exc:
                logger.error("배터리 읽기 오류: %s", exc)

            time.sleep(1.0)

    def _read_ina219(self) -> None:
        """INA219 센서에서 데이터를 읽는다."""
        voltage = self._sensor.voltage()
        current_ma = self._sensor.current()
        power_mw = self._sensor.power()

        self._update_status(voltage, current_ma, power_mw)

    def _read_mock(self) -> None:
        """Mock 데이터를 생성한다."""
        with self._lock:
            # 시뮬레이션 값 사용
            voltage = self._min_voltage + (self._max_voltage - self._min_voltage) * (
                self._mock_soc / 100
            )
            current_ma = self._mock_current
            power_mw = abs(voltage * current_ma)

        self._update_status(voltage, current_ma, power_mw, mock_soc=self._mock_soc)

    def _update_status(
        self,
        voltage: float,
        current_ma: float,
        power_mw: float,
        mock_soc: Optional[float] = None,
    ) -> None:
        """상태를 업데이트한다."""
        # SoC 계산
        if mock_soc is not None:
            soc = mock_soc
        else:
            # 전압 기반 SoC 추정 (선형)
            if voltage <= self._min_voltage:
                soc = 0.0
            elif voltage >= self._max_voltage:
                soc = 100.0
            else:
                soc = (
                    (voltage - self._min_voltage)
                    / (self._max_voltage - self._min_voltage)
                ) * 100.0

        # 충전 감지 (음수 전류 = 충전)
        is_charging = current_ma < -50  # 50mA 이상 역전류

        # 저전력/긴급 판정
        is_low = soc <= self._low_threshold
        is_critical = soc <= self._critical_threshold

        with self._lock:
            self._latest_status = BatteryStatus(
                battery_type=self._battery_type,
                voltage=voltage,
                current_ma=current_ma,
                power_mw=power_mw,
                soc=soc,
                is_charging=is_charging,
                is_low=is_low,
                is_critical=is_critical,
                timestamp=time.time(),
            )


# ---------------------------------------------------------------------------
# Dual Battery Manager
# ---------------------------------------------------------------------------


class DualBatteryManager:
    """
    로봇 배터리 + 출력 배터리를 함께 관리하는 클래스.
    """

    def __init__(self, config: dict) -> None:
        robot_cfg = config.get("robot_battery", {})
        output_cfg = config.get("output_battery", {})

        self.robot_battery = BatteryMonitor(robot_cfg, BatteryType.ROBOT)
        self.output_battery = BatteryMonitor(output_cfg, BatteryType.OUTPUT)

        logger.info("DualBatteryManager 초기화 완료")

    def start(self) -> None:
        """모든 배터리 모니터링을 시작한다."""
        self.robot_battery.start()
        self.output_battery.start()

    def stop(self) -> None:
        """모든 배터리 모니터링을 중지한다."""
        self.robot_battery.stop()
        self.output_battery.stop()

    def get_robot_status(self) -> BatteryStatus:
        """로봇 배터리 상태를 반환한다."""
        return self.robot_battery.get_status()

    def get_output_status(self) -> BatteryStatus:
        """출력 배터리 상태를 반환한다."""
        return self.output_battery.get_status()

    def is_robot_battery_low(self) -> bool:
        """로봇 배터리 저전력 여부."""
        return self.robot_battery.is_low_battery()

    def is_output_charging(self) -> bool:
        """출력 배터리 충전 중 여부 (= 기기에 전력 공급 중)."""
        # 출력 배터리에서 전류가 나가면 기기 충전 중
        status = self.output_battery.get_status()
        return status.current_ma > 50  # 50mA 이상 출력

    def get_output_current_ma(self) -> float:
        """출력 전류 (mA)."""
        return self.output_battery.get_current_ma()

    def get_output_soc(self) -> float:
        """출력 배터리 SoC (%)."""
        return self.output_battery.get_soc()
