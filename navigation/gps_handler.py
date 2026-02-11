"""
Bridge Guardian — GPS 수신 및 파싱 모듈
pynmea2를 사용한 NMEA ($GPGGA/$GPRMC) 문장 처리, 백그라운드 스레드 기반.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GPSData:
    """GPS 위치 데이터"""

    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed_kmh: float = 0.0          # km/h (GPRMC 기반)
    heading: float = 0.0            # 진행 방향 (도)
    fix_quality: int = 0            # 0=없음, 1=GPS, 2=DGPS
    num_satellites: int = 0
    timestamp: float = 0.0          # time.time() 기준
    valid: bool = False

    @property
    def has_fix(self) -> bool:
        """유효한 GPS 수신 여부"""
        return self.fix_quality > 0 and self.valid


# ---------------------------------------------------------------------------
# GPSHandler
# ---------------------------------------------------------------------------

class GPSHandler:
    """
    GPS 시리얼 수신기.

    백그라운드 스레드에서 NMEA 문장을 파싱하고 최신 위치를 유지한다.
    pynmea2 / serial이 없는 환경에서도 생성 가능 (graceful degradation).

    Parameters
    ----------
    port : str
        시리얼 포트 (예: 'COM3', '/dev/ttyUSB0').
    baudrate : int
        보드레이트 (기본 9600).
    update_interval : float
        최소 갱신 주기 (초).
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        update_interval: float = 1.0,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._update_interval = update_interval

        self._lock = threading.Lock()
        self._latest = GPSData()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial = None

        # 선택 의존성 로드
        self._pynmea2 = self._try_import("pynmea2")
        self._serial_mod = self._try_import("serial")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_import(module_name: str):
        """모듈을 선택적으로 임포트한다. 실패하면 None 반환."""
        try:
            import importlib
            return importlib.import_module(module_name)
        except ImportError:
            logger.warning("%s 모듈을 찾을 수 없습니다. GPS 기능이 제한됩니다.", module_name)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """GPS 수신 백그라운드 스레드를 시작한다."""
        if self._serial_mod is None or self._pynmea2 is None:
            logger.error("pynmea2 또는 pyserial이 설치되지 않아 GPS를 시작할 수 없습니다.")
            return False

        if self._running:
            logger.warning("GPS 스레드가 이미 실행 중입니다.")
            return True

        try:
            self._serial = self._serial_mod.Serial(
                self._port, self._baudrate, timeout=1,
            )
            time.sleep(0.5)  # 시리얼 안정화 대기
            logger.info("GPS 시리얼 포트 열림: %s @ %d", self._port, self._baudrate)
        except Exception:
            logger.exception("GPS 시리얼 포트 열기 실패: %s", self._port)
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, name="gps-reader", daemon=True,
        )
        self._thread.start()
        logger.info("GPS 수신 스레드 시작됨")
        return True

    def stop(self) -> None:
        """GPS 수신 스레드를 종료한다."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        logger.info("GPS 수신 스레드 종료됨")

    def get_current(self) -> GPSData:
        """최신 GPS 데이터 사본을 반환한다."""
        with self._lock:
            return GPSData(
                latitude=self._latest.latitude,
                longitude=self._latest.longitude,
                altitude=self._latest.altitude,
                speed_kmh=self._latest.speed_kmh,
                heading=self._latest.heading,
                fix_quality=self._latest.fix_quality,
                num_satellites=self._latest.num_satellites,
                timestamp=self._latest.timestamp,
                valid=self._latest.valid,
            )

    def get_location_string(self) -> str:
        """
        현재 위치를 한국어 형식 문자열로 반환한다.

        Returns
        -------
        str
            예: ``"위도 37.5142°N, 경도 126.9123°E (위성 8개, 고도 15.2m)"``
            GPS 수신 불량 시: ``"GPS 수신 대기 중…"``
        """
        data = self.get_current()
        if not data.has_fix:
            return "GPS 수신 대기 중…"

        ns = "N" if data.latitude >= 0 else "S"
        ew = "E" if data.longitude >= 0 else "W"
        return (
            f"위도 {abs(data.latitude):.4f}°{ns}, "
            f"경도 {abs(data.longitude):.4f}°{ew} "
            f"(위성 {data.num_satellites}개, 고도 {data.altitude:.1f}m)"
        )

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """시리얼 포트에서 NMEA 문장을 읽어 파싱한다 (백그라운드)."""
        while self._running:
            try:
                if self._serial is None or not self._serial.is_open:
                    time.sleep(self._update_interval)
                    continue

                raw_line = self._serial.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("ascii", errors="replace").strip()
                if not line:
                    continue

                self._parse_nmea(line)

            except Exception:
                logger.debug("GPS 읽기 오류 (재시도 중)", exc_info=True)
                time.sleep(self._update_interval)

    def _parse_nmea(self, sentence: str) -> None:
        """NMEA 문장을 파싱하여 내부 상태를 갱신한다."""
        pynmea2 = self._pynmea2
        if pynmea2 is None:
            return

        try:
            msg = pynmea2.parse(sentence)
        except Exception:
            return  # 알 수 없는 / 손상된 문장은 무시

        now = time.time()

        with self._lock:
            if sentence.startswith("$GPGGA") or sentence.startswith("$GNGGA"):
                self._parse_gga(msg, now)
            elif sentence.startswith("$GPRMC") or sentence.startswith("$GNRMC"):
                self._parse_rmc(msg, now)

    def _parse_gga(self, msg, now: float) -> None:
        """$GPGGA 문장 처리 — 위치, 고도, 위성 수, 수신 품질."""
        try:
            lat = msg.latitude if msg.latitude else 0.0
            lon = msg.longitude if msg.longitude else 0.0
            self._latest.latitude = float(lat)
            self._latest.longitude = float(lon)
            self._latest.altitude = float(msg.altitude) if msg.altitude else 0.0
            self._latest.fix_quality = int(msg.gps_qual) if msg.gps_qual else 0
            self._latest.num_satellites = int(msg.num_sats) if msg.num_sats else 0
            self._latest.timestamp = now
            self._latest.valid = self._latest.fix_quality > 0
        except (AttributeError, ValueError, TypeError):
            logger.debug("GGA 파싱 실패", exc_info=True)

    def _parse_rmc(self, msg, now: float) -> None:
        """$GPRMC 문장 처리 — 속도, 방향, 위치 (GGA 보완)."""
        try:
            if hasattr(msg, "status") and msg.status == "A":
                lat = msg.latitude if msg.latitude else 0.0
                lon = msg.longitude if msg.longitude else 0.0
                self._latest.latitude = float(lat)
                self._latest.longitude = float(lon)
                self._latest.valid = True

            # 속도: 노트 → km/h
            if msg.spd_over_grnd is not None:
                self._latest.speed_kmh = float(msg.spd_over_grnd) * 1.852

            # 진행 방향
            if msg.true_course is not None:
                self._latest.heading = float(msg.true_course)

            self._latest.timestamp = now
        except (AttributeError, ValueError, TypeError):
            logger.debug("RMC 파싱 실패", exc_info=True)
