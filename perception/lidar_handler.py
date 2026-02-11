"""
Bridge Guardian — LiDAR Handler
2D LiDAR (RPLidar A1/A2) scan acquisition and object detection.

Supports two modes:
  - "rplidar": Live capture via rplidar package or serial fallback.
  - "dummy":   Synthetic scans for development without hardware.

Inspired by H-Mobility lidar_perception_pkg patterns (360-degree scan,
angle/distance indexing, sector-based detection, stability filtering).
"""

from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Graceful rplidar import — try pip package first, then serial fallback flag
# ---------------------------------------------------------------------------
_RPLIDAR_AVAILABLE = False
_SERIAL_AVAILABLE = False

try:
    from rplidar import RPLidar, RPLidarException

    _RPLIDAR_AVAILABLE = True
except ImportError:
    RPLidar = None  # type: ignore
    RPLidarException = Exception  # type: ignore
    logger.warning("rplidar package not installed — trying serial fallback")

if not _RPLIDAR_AVAILABLE:
    try:
        import serial

        _SERIAL_AVAILABLE = True
    except ImportError:
        serial = None  # type: ignore
        logger.warning(
            "pyserial not installed — LiDAR hardware unavailable, dummy mode only"
        )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class LidarScan:
    """Container for a single 360-degree LiDAR scan."""

    ranges: List[float] = field(default_factory=lambda: [float("inf")] * 360)
    intensities: List[float] = field(default_factory=lambda: [0.0] * 360)
    timestamp: float = 0.0
    valid: bool = False


@dataclass
class LidarDetection:
    """A detected object cluster from LiDAR data."""

    angle_deg: float  # centre angle of detection [0, 360)
    distance_m: float  # distance from sensor in metres
    cluster_size: int  # number of consecutive scan points in cluster
    is_person_candidate: bool  # heuristic: cluster width consistent with a person


# ---------------------------------------------------------------------------
# LidarHandler
# ---------------------------------------------------------------------------
class LidarHandler:
    """2D LiDAR acquisition and object detection with background scanning thread.

    Parameters
    ----------
    config : dict
        Optional keys:
        - ``port``           (str)   – serial port, e.g. ``"COM3"`` / ``"/dev/ttyUSB0"``
        - ``baudrate``       (int)   – serial baudrate (default 115200)
        - ``mode``           (str)   – ``"rplidar"`` | ``"dummy"`` (default auto-detect)
        - ``range_min_m``    (float) – minimum valid range in metres (default 0.15)
        - ``range_max_m``    (float) – maximum valid range in metres (default 12.0)
        - ``scan_rate_hz``   (float) – target scan rate for dummy mode (default 10.0)
        - ``cluster_min_points`` (int) – min consecutive points for a cluster (default 3)
        - ``person_width_min_m`` (float) – min cluster arc width for person (default 0.2)
        - ``person_width_max_m`` (float) – max cluster arc width for person (default 1.2)
    """

    def __init__(self, config: dict) -> None:
        self._port: str = config.get("port", "/dev/ttyUSB0")
        self._baudrate: int = config.get("baudrate", 115200)
        self._range_min: float = config.get("range_min_m", 0.15)
        self._range_max: float = config.get("range_max_m", 12.0)
        self._scan_rate: float = config.get("scan_rate_hz", 10.0)
        self._cluster_min: int = config.get("cluster_min_points", 3)
        self._person_w_min: float = config.get("person_width_min_m", 0.2)
        self._person_w_max: float = config.get("person_width_max_m", 1.2)

        # Determine operating mode
        requested_mode: str = config.get("mode", "auto")
        if requested_mode == "dummy":
            self._mode = "dummy"
        elif requested_mode in ("rplidar", "auto"):
            if _RPLIDAR_AVAILABLE:
                self._mode = "rplidar"
            elif _SERIAL_AVAILABLE:
                self._mode = "serial"
            else:
                self._mode = "dummy"
        else:
            self._mode = "dummy"

        # State
        self._lidar = None
        self._scan_generator = None
        self._latest_scan: LidarScan = LidarScan()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(
            "LidarHandler created (mode=%s, port=%s, range=[%.2f, %.2f]m)",
            self._mode,
            self._port,
            self._range_min,
            self._range_max,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current operating mode: 'rplidar', 'serial', or 'dummy'."""
        return self._mode

    @property
    def is_running(self) -> bool:
        """True when background scan thread is active."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Initialise hardware and start the background scan thread.

        Returns
        -------
        bool
            True if started successfully, False otherwise.
        """
        if self._running:
            logger.warning("LidarHandler already running")
            return True

        if self._mode == "rplidar":
            self._init_rplidar()
        elif self._mode == "serial":
            self._init_serial()
        else:
            logger.info("LiDAR running in dummy mode — no hardware")

        self._running = True
        self._thread = threading.Thread(
            target=self._scan_loop, daemon=True, name="lidar-scan"
        )
        self._thread.start()
        logger.info("LiDAR background scan thread started (mode=%s)", self._mode)
        return True

    def stop(self) -> None:
        """Stop scanning and release hardware resources."""
        if not self._running:
            return
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        self._shutdown_hardware()
        logger.info("LidarHandler stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scan(self) -> LidarScan:
        """Return a *copy* of the latest 360-degree scan (thread-safe)."""
        with self._lock:
            return LidarScan(
                ranges=list(self._latest_scan.ranges),
                intensities=list(self._latest_scan.intensities),
                timestamp=self._latest_scan.timestamp,
                valid=self._latest_scan.valid,
            )

    def detect_objects(self, scan: Optional[LidarScan] = None) -> List[LidarDetection]:
        """Cluster-based object detection on a 360-degree scan.

        Parameters
        ----------
        scan : LidarScan, optional
            Scan to analyse.  If ``None`` the latest internal scan is used.

        Returns
        -------
        List[LidarDetection]
            Detected object clusters sorted by distance (nearest first).
        """
        if scan is None:
            scan = self.get_scan()
        if not scan.valid:
            return []

        # --- find clusters of consecutive in-range points ---
        clusters: List[List[int]] = []
        current_cluster: List[int] = []

        for i in range(360):
            r = scan.ranges[i]
            if self._range_min <= r <= self._range_max:
                current_cluster.append(i)
            else:
                if len(current_cluster) >= self._cluster_min:
                    clusters.append(current_cluster)
                current_cluster = []

        # Handle wrap-around (0° / 359°)
        if len(current_cluster) >= self._cluster_min:
            if clusters and clusters[0][0] == 0:
                # Merge with first cluster if it starts at 0°
                clusters[0] = current_cluster + clusters[0]
            else:
                clusters.append(current_cluster)

        # --- convert clusters to detections ---
        detections: List[LidarDetection] = []
        for cluster in clusters:
            distances = [scan.ranges[i] for i in cluster]
            mean_dist = float(np.mean(distances))
            min_dist = float(np.min(distances))

            # Centre angle (handle wrap-around)
            angles = np.array(cluster, dtype=float)
            if angles[-1] - angles[0] > 180:
                # Cluster wraps around 0°
                angles = np.where(angles < 180, angles + 360, angles)
            centre_angle = float(np.mean(angles)) % 360.0

            # Estimate cluster arc width at mean distance
            arc_deg = len(cluster)  # 1 point ≈ 1°
            arc_width_m = mean_dist * math.radians(arc_deg)

            is_person = self._person_w_min <= arc_width_m <= self._person_w_max

            detections.append(
                LidarDetection(
                    angle_deg=centre_angle,
                    distance_m=min_dist,
                    cluster_size=len(cluster),
                    is_person_candidate=is_person,
                )
            )

        detections.sort(key=lambda d: d.distance_m)
        return detections

    def get_nearest_in_sector(
        self,
        centre_deg: float,
        width_deg: float,
        scan: Optional[LidarScan] = None,
    ) -> Optional[float]:
        """Return the nearest valid range (metres) within a sector, or None.

        Parameters
        ----------
        centre_deg : float
            Centre angle of the sector in degrees [0, 360).
        width_deg : float
            Total angular width of the sector in degrees.
        scan : LidarScan, optional
            Scan to query.  Defaults to latest internal scan.
        """
        if scan is None:
            scan = self.get_scan()
        if not scan.valid:
            return None

        half = width_deg / 2.0
        start = (centre_deg - half) % 360
        end = (centre_deg + half) % 360

        nearest = float("inf")
        for i in range(360):
            # Check if angle i is inside [start, end] (handling wrap)
            if start <= end:
                in_sector = start <= i <= end
            else:
                in_sector = i >= start or i <= end

            if in_sector:
                r = scan.ranges[i]
                if self._range_min <= r <= self._range_max and r < nearest:
                    nearest = r

        return nearest if nearest < float("inf") else None

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_rplidar(self) -> None:
        """Initialise via the rplidar pip package."""
        try:
            self._lidar = RPLidar(self._port, baudrate=self._baudrate)
            info = self._lidar.get_info()
            health = self._lidar.get_health()
            logger.info(
                "RPLidar connected — model=%s, firmware=%s, health=%s",
                info.get("model", "?"),
                info.get("firmware", "?"),
                health[0],
            )
            self._scan_generator = self._lidar.iter_scans()
        except Exception as exc:
            logger.error(
                "Failed to initialise RPLidar: %s — falling back to dummy mode", exc
            )
            self._mode = "dummy"
            self._lidar = None

    def _init_serial(self) -> None:
        """Minimal serial fallback (placeholder for custom protocol)."""
        try:
            self._lidar = serial.Serial(
                self._port,
                self._baudrate,
                timeout=1.0,
            )
            logger.info("Serial LiDAR connection opened on %s", self._port)
        except Exception as exc:
            logger.error(
                "Failed to open serial port %s: %s — falling back to dummy mode",
                self._port,
                exc,
            )
            self._mode = "dummy"
            self._lidar = None

    def _shutdown_hardware(self) -> None:
        """Release hardware resources safely."""
        if self._lidar is None:
            return
        try:
            if self._mode == "rplidar" and _RPLIDAR_AVAILABLE:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            elif self._mode == "serial" and hasattr(self._lidar, "close"):
                self._lidar.close()
            logger.info("LiDAR hardware released")
        except Exception as exc:
            logger.warning("Error releasing LiDAR hardware: %s", exc)
        finally:
            self._lidar = None

    # ------------------------------------------------------------------
    # Background scan loop
    # ------------------------------------------------------------------

    def _scan_loop(self) -> None:
        """Background thread: continuously read scans and update latest."""
        while self._running:
            try:
                if self._mode == "rplidar":
                    self._read_rplidar_scan()
                elif self._mode == "serial":
                    self._read_serial_scan()
                else:
                    self._generate_dummy_scan()
            except Exception as exc:
                logger.error("LiDAR scan error: %s", exc)
                time.sleep(0.5)

    def _read_rplidar_scan(self) -> None:
        """Read one full 360° scan from rplidar package (H-Mobility pattern)."""
        if self._scan_generator is None:
            time.sleep(0.1)
            return

        try:
            raw_scan = next(self._scan_generator)
        except StopIteration:
            logger.warning("RPLidar scan generator ended — reinitialising")
            self._init_rplidar()
            return

        # Convert raw measurements → 360-element arrays
        # raw_scan is list of (quality, angle, distance_mm)
        ranges = [float("inf")] * 360
        intensities = [0.0] * 360

        for quality, angle, distance in raw_scan:
            idx = int(round(angle)) % 360
            dist_m = distance / 1000.0  # mm → m
            if dist_m > 0:
                # Keep nearest reading per degree bucket (H-Mobility pattern)
                if dist_m < ranges[idx]:
                    ranges[idx] = dist_m
                    intensities[idx] = float(quality)

        scan = LidarScan(
            ranges=ranges,
            intensities=intensities,
            timestamp=time.time(),
            valid=True,
        )
        with self._lock:
            self._latest_scan = scan

    def _read_serial_scan(self) -> None:
        """Placeholder for custom serial protocol reading."""
        # Serial fallback: sleep and generate dummy for now
        # Override this method for a specific serial LiDAR protocol
        self._generate_dummy_scan()

    def _generate_dummy_scan(self) -> None:
        """Generate a synthetic scan with 2-3 simulated objects for testing."""
        ranges = [float("inf")] * 360
        intensities = [0.0] * 360

        # Simulate objects at known positions for development
        # Object 1: person-sized at ~3m, 45° (cluster width ~10°)
        for i in range(40, 50):
            ranges[i] = 3.0 + np.random.uniform(-0.1, 0.1)
            intensities[i] = 50.0

        # Object 2: person-sized at ~1.5m, 180° (cluster width ~15°)
        for i in range(173, 188):
            ranges[i] = 1.5 + np.random.uniform(-0.05, 0.05)
            intensities[i] = 70.0

        # Object 3: small object at ~5m, 300°
        for i in range(299, 302):
            ranges[i] = 5.0 + np.random.uniform(-0.2, 0.2)
            intensities[i] = 30.0

        scan = LidarScan(
            ranges=ranges,
            intensities=intensities,
            timestamp=time.time(),
            valid=True,
        )
        with self._lock:
            self._latest_scan = scan

        time.sleep(1.0 / self._scan_rate)
