#!/usr/bin/env python3
"""
ChargeMate - 자율주행 이동형 충전 로봇 메인 파이프라인

모든 모듈을 통합하여 실행하는 진입점.
FastAPI 서버 + 로봇 제어 루프를 병렬로 실행한다.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# --- Project modules ---
from utils.logger import get_logger
from perception.realsense_handler import CameraHandler
from perception.person_detector import PersonDetector
from perception.person_tracker import PersonTracker
from navigation.gps_handler import GPSHandler
from navigation.motor_controller import MotorController
from navigation.path_planner import PathPlanner
from charging.battery_monitor import DualBatteryManager
from user_detection.hand_detector import HandDetector
from behavior.state_machine import ChargingStateMachine, RobotState
from communication.tts_service import TTSService
from app_backend.api import (
    app,
    queue_manager,
    update_robot_state,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global shutdown flag
# ---------------------------------------------------------------------------
_shutdown_requested = threading.Event()


def signal_handler(signum, frame):
    """SIGINT/SIGTERM 처리"""
    logger.info("종료 신호 수신: %s", signal.Signals(signum).name)
    _shutdown_requested.set()


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """YAML 설정 파일을 로드한다."""
    path = Path(config_path)
    if not path.exists():
        logger.error("설정 파일을 찾을 수 없습니다: %s", config_path)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("설정 로드 완료: %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Robot Controller
# ---------------------------------------------------------------------------


class ChargeMateController:
    """
    ChargeMate 로봇 통합 컨트롤러.

    모든 서브시스템을 초기화하고 메인 루프를 실행한다.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._running = False

        # --- Subsystems ---
        self.camera: Optional[CameraHandler] = None
        self.detector: Optional[PersonDetector] = None
        self.tracker: Optional[PersonTracker] = None
        self.gps: Optional[GPSHandler] = None
        self.motor: Optional[MotorController] = None
        self.path_planner: Optional[PathPlanner] = None
        self.battery_manager: Optional[DualBatteryManager] = None
        self.hand_detector: Optional[HandDetector] = None
        self.state_machine: Optional[ChargingStateMachine] = None
        self.tts: Optional[TTSService] = None

        # --- State ---
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0

        logger.info("ChargeMateController 생성됨")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """모든 서브시스템을 초기화한다."""
        try:
            # 1. Camera
            logger.info("카메라 초기화 중...")
            self.camera = CameraHandler(self.config.get("camera", {}))

            # 2. Person Detector (YOLOv8-pose)
            logger.info("사람 감지기 초기화 중...")
            det_cfg = self.config.get("detection", {})
            self.detector = PersonDetector(
                {
                    "model_path": det_cfg.get("model_path", "yolov8n-pose.pt"),
                    "confidence": det_cfg.get("confidence", 0.5),
                    "iou_threshold": det_cfg.get("iou_threshold", 0.45),
                    "device": det_cfg.get("device", None),
                    "input_size": det_cfg.get("input_size", 640),
                    "tracker_config": det_cfg.get("tracker", "botsort.yaml"),
                }
            )

            # 3. Person Tracker
            logger.info("사람 추적기 초기화 중...")
            self.tracker = PersonTracker(self.config.get("tracking", {}))

            # 4. GPS Handler
            gps_cfg = self.config.get("gps", {})
            if gps_cfg.get("enabled", False):
                logger.info("GPS 핸들러 초기화 중...")
                self.gps = GPSHandler(
                    port=gps_cfg.get("port", "COM4"),
                    baudrate=gps_cfg.get("baudrate", 9600),
                    update_interval=1.0 / gps_cfg.get("update_rate_hz", 1.0),
                )

            # 5. Motor Controller
            motor_cfg = self.config.get("motor", {})
            if motor_cfg.get("enabled", False):
                logger.info("모터 컨트롤러 초기화 중...")
                self.motor = MotorController(
                    port=motor_cfg.get("port", "COM5"),
                    baudrate=motor_cfg.get("baudrate", 9600),
                )

            # 6. Path Planner
            logger.info("경로 계획기 초기화 중...")
            robot_cfg = self.config.get("robot", {})
            self.path_planner = PathPlanner(
                arrival_radius_m=robot_cfg.get("arrival_radius_m", 3.0),
                approach_speed=motor_cfg.get("approach_speed", 50),
                patrol_speed=motor_cfg.get("patrol_speed", 80),
            )

            # 7. Battery Manager
            logger.info("배터리 매니저 초기화 중...")
            self.battery_manager = DualBatteryManager(self.config.get("battery", {}))

            # 8. Hand Detector
            logger.info("손 감지기 초기화 중...")
            self.hand_detector = HandDetector(self.config.get("hand_detection", {}))

            # 9. State Machine
            logger.info("상태 머신 초기화 중...")
            behavior_cfg = self.config.get("behavior", {})
            behavior_cfg["home_latitude"] = robot_cfg.get("home_latitude", 37.5665)
            behavior_cfg["home_longitude"] = robot_cfg.get("home_longitude", 126.9780)
            self.state_machine = ChargingStateMachine(behavior_cfg)

            # 10. TTS Service
            logger.info("TTS 서비스 초기화 중...")
            self.tts = TTSService(self.config.get("tts", {}))

            logger.info("모든 서브시스템 초기화 완료")
            return True

        except Exception as exc:
            logger.exception("서브시스템 초기화 실패: %s", exc)
            return False

    def start(self) -> None:
        """서브시스템들을 시작한다."""
        if self.gps:
            self.gps.start()

        if self.motor:
            self.motor.connect()

        if self.battery_manager:
            self.battery_manager.start()

        self._running = True
        logger.info("ChargeMateController 시작됨")

    def stop(self) -> None:
        """모든 서브시스템을 정지한다."""
        self._running = False

        # 모터 정지
        if self.motor:
            self.motor.stop()
            self.motor.disconnect()

        # GPS 정지
        if self.gps:
            self.gps.stop()

        # 배터리 모니터링 정지
        if self.battery_manager:
            self.battery_manager.stop()

        # 손 감지기 정지
        if self.hand_detector:
            self.hand_detector.stop()

        # TTS 정지
        if self.tts:
            self.tts.stop()

        # 카메라 정지
        if self.camera:
            self.camera.stop()

        logger.info("ChargeMateController 정지됨")

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """메인 제어 루프를 실행한다."""
        logger.info("메인 루프 시작")

        while self._running and not _shutdown_requested.is_set():
            try:
                self._process_frame()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.exception("프레임 처리 오류: %s", exc)
                time.sleep(0.1)

        logger.info("메인 루프 종료")

    def _process_frame(self) -> None:
        """단일 프레임을 처리한다."""
        # 1. 카메라 프레임 획득
        frame_data = self.camera.get_frames()
        if frame_data is None:
            time.sleep(0.01)
            return

        frame = frame_data.color_image
        depth_frame = frame_data.depth_frame
        timestamp = frame_data.timestamp

        # 2. 사람 감지 + 추적
        detections = self.detector.detect(frame)
        tracked_persons = self.tracker.update(
            detections,
            timestamp,
            depth_frame=depth_frame,
            camera_handler=self.camera,
        )

        # 3. 손 흔들기 감지
        waving_persons = []
        if self.hand_detector.is_enabled:
            waving_persons = self.hand_detector.update_waving_detection(
                frame,
                tracked_persons,
                camera_handler=self.camera,
                depth_frame=depth_frame,
            )

        # 4. GPS / 배터리 상태 수집
        gps_data = self.gps.get_current() if self.gps else None
        current_lat = gps_data.latitude if gps_data and gps_data.has_fix else 0.0
        current_lon = gps_data.longitude if gps_data and gps_data.has_fix else 0.0

        robot_battery_low = False
        output_soc = 0.0
        is_charging = False
        charging_current_ma = 0.0

        if self.battery_manager:
            robot_battery_low = self.battery_manager.is_robot_battery_low()
            output_soc = self.battery_manager.get_output_soc()
            is_charging = self.battery_manager.is_output_charging()
            charging_current_ma = self.battery_manager.get_output_current_ma()

        # 5. 대기열 요청 체크 (IDLE 상태에서만)
        if self.state_machine.state == RobotState.IDLE:
            pending = queue_manager.get_next_pending()
            if pending:
                accepted = self.state_machine.accept_request(
                    request_id=pending.request_id,
                    latitude=pending.latitude,
                    longitude=pending.longitude,
                    user_name=pending.user_name,
                )
                if accepted:
                    queue_manager.accept_request(pending.request_id)
                    logger.info("요청 수락: %s", pending.request_id)

        # 6. 거리 계산
        distance_to_target = float("inf")
        distance_to_home = float("inf")
        navigation_steering = 0
        navigation_speed = 0

        home_lat, home_lon = self.state_machine.home_position

        if current_lat != 0.0 and current_lon != 0.0:
            # 목표까지 거리
            status = self.state_machine.get_status()
            target_lat = status.get("target_latitude", 0)
            target_lon = status.get("target_longitude", 0)

            if target_lat and target_lon:
                approach = self.path_planner.plan_approach(
                    current_lat, current_lon, target_lat, target_lon
                )
                distance_to_target = approach["distance_m"]
                navigation_steering = approach["steering"]
                navigation_speed = approach["speed"]

            # 홈까지 거리
            home_approach = self.path_planner.plan_approach(
                current_lat, current_lon, home_lat, home_lon
            )
            distance_to_home = home_approach["distance_m"]

            # 복귀 중이면 홈 방향으로 네비게이션
            if self.state_machine.state == RobotState.RETURNING:
                navigation_steering = home_approach["steering"]
                navigation_speed = home_approach["speed"]

        # 7. 손 흔드는 사용자 정보
        detected_waving_user = self.hand_detector.get_waving_person_dict()

        # 8. 상태 머신 업데이트
        action = self.state_machine.update(
            current_lat=current_lat,
            current_lon=current_lon,
            distance_to_target=distance_to_target,
            distance_to_home=distance_to_home,
            detected_waving_user=detected_waving_user,
            is_charging=is_charging,
            charging_current_ma=charging_current_ma,
            output_soc=output_soc,
            robot_battery_low=robot_battery_low,
            navigation_steering=navigation_steering,
            navigation_speed=navigation_speed,
        )

        # 9. 모터 명령 전송
        if self.motor and action.get("motor_command"):
            self.motor.send(action["motor_command"])

        # 10. TTS 메시지 재생
        if self.tts and action.get("tts_message"):
            self.tts.speak_text(action["tts_message"], priority=3)

        # 11. API 상태 업데이트
        robot_status = self.state_machine.get_status()
        update_robot_state(
            status=robot_status["state"],
            latitude=current_lat,
            longitude=current_lon,
            robot_soc=self.battery_manager.robot_battery.get_soc()
            if self.battery_manager
            else 0,
            output_soc=output_soc,
            current_request_id=robot_status.get("current_request_id"),
            state_duration_s=robot_status.get("state_duration_s", 0),
            is_output_charging=is_charging,
            output_current_ma=charging_current_ma,
        )

        # 12. 요청 완료 처리
        if (
            self.state_machine.previous_state == RobotState.UNDOCKING
            and self.state_machine.state == RobotState.RETURNING
        ):
            request_id = robot_status.get("current_request_id")
            if request_id:
                queue_manager.complete_request(
                    request_id,
                    charging_duration_s=robot_status.get("state_duration_s", 0),
                    energy_delivered_wh=0,  # TODO: 실제 충전량 계산
                )

        # 13. 시각화 (옵션)
        vis_cfg = self.config.get("visualization", {})
        if vis_cfg.get("show_camera", True):
            self._visualize(frame, tracked_persons, waving_persons, robot_status)

        # 14. FPS 계산
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now

    def _visualize(
        self,
        frame: np.ndarray,
        tracked_persons: list,
        waving_persons: list,
        robot_status: dict,
    ) -> None:
        """프레임에 오버레이를 그리고 표시한다."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 감지된 사람 그리기
        for person in tracked_persons:
            x1, y1, x2, y2 = person.bbox.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID:{person.track_id}"
            if person.depth_m:
                label += f" {person.depth_m:.1f}m"

            cv2.putText(
                vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # 손 흔드는 사람 표시
        for wp in waving_persons:
            x1, y1, x2, y2 = wp.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(
                vis,
                f"WAVING! ({wp.wave_count})",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # 상태 오버레이
        state_text = f"State: {robot_status['state']}"
        cv2.putText(
            vis, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        fps_text = f"FPS: {self._fps:.1f}"
        cv2.putText(
            vis, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        if robot_status.get("current_request_id"):
            req_text = f"Request: {robot_status['current_request_id'][:8]}..."
            cv2.putText(
                vis, req_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        window_name = self.config.get("visualization", {}).get(
            "window_name", "ChargeMate"
        )
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q or ESC
            _shutdown_requested.set()
        elif key == ord("e"):  # Emergency stop
            self.state_machine.emergency_stop()
            logger.warning("비상 정지 활성화 (키보드)")


# ---------------------------------------------------------------------------
# API Server Thread
# ---------------------------------------------------------------------------


def run_api_server(host: str, port: int) -> None:
    """별도 스레드에서 FastAPI 서버를 실행한다."""
    import asyncio
    import uvicorn

    async def serve():
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(serve())


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """메인 진입점."""
    parser = argparse.ArgumentParser(
        description="ChargeMate - 자율주행 이동형 충전 로봇"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/settings.yaml",
        help="설정 파일 경로 (기본: config/settings.yaml)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="GUI 시각화 비활성화",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="API 서버만 실행 (로봇 제어 없음)",
    )
    args = parser.parse_args()

    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 설정 로드
    config = load_config(args.config)

    if args.no_gui:
        config.setdefault("visualization", {})["show_camera"] = False

    # API 서버 시작
    api_cfg = config.get("api", {})
    api_host = api_cfg.get("host", "0.0.0.0")
    api_port = api_cfg.get("port", 8000)

    api_thread = threading.Thread(
        target=run_api_server,
        args=(api_host, api_port),
        name="api-server",
        daemon=True,
    )
    api_thread.start()
    logger.info("API 서버 시작: http://%s:%d", api_host, api_port)

    if args.api_only:
        logger.info("API 전용 모드 - 로봇 제어 비활성화")
        try:
            while not _shutdown_requested.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        logger.info("종료")
        return

    # 로봇 컨트롤러 초기화 및 실행
    controller = ChargeMateController(config)

    if not controller.initialize():
        logger.error("초기화 실패 - 종료")
        sys.exit(1)

    try:
        controller.start()
        controller.run()
    except KeyboardInterrupt:
        logger.info("키보드 인터럽트")
    finally:
        controller.stop()
        cv2.destroyAllWindows()
        logger.info("ChargeMate 종료")


if __name__ == "__main__":
    main()
