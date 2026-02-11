"""
ChargeMate - FastAPI 백엔드 서버
충전 요청 관리 및 로봇 상태 API.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from utils.logger import get_logger
from app_backend.models import (
    ChargeRequest,
    ChargeRequestResponse,
    ChargeRequestDetail,
    RequestStatus,
    RobotStatus,
    RobotStatusResponse,
    RobotPosition,
    BatteryInfo,
    QueueStatus,
    WSRobotUpdate,
    WSRequestUpdate,
    generate_request_id,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request Queue Manager
# ---------------------------------------------------------------------------


class RequestQueueManager:
    """충전 요청 대기열 관리자"""

    def __init__(self, max_queue_size: int = 10) -> None:
        self._max_size = max_queue_size
        self._requests: Dict[str, ChargeRequestDetail] = {}
        self._pending_queue: List[str] = []  # 대기 순서
        self._completed_today: int = 0
        self._start_of_day: float = self._get_start_of_day()

    @staticmethod
    def _get_start_of_day() -> float:
        """오늘 시작 타임스탬프"""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()

    def add_request(self, req: ChargeRequest) -> ChargeRequestDetail:
        """새 요청 추가"""
        # 날짜 체크
        if time.time() > self._start_of_day + 86400:
            self._completed_today = 0
            self._start_of_day = self._get_start_of_day()

        # 대기열 크기 체크
        if len(self._pending_queue) >= self._max_size:
            raise HTTPException(
                status_code=503,
                detail=f"대기열이 가득 찼습니다. (최대 {self._max_size}개)",
            )

        request_id = generate_request_id()
        now = datetime.now()

        detail = ChargeRequestDetail(
            request_id=request_id,
            status=RequestStatus.PENDING,
            latitude=req.latitude,
            longitude=req.longitude,
            user_name=req.user_name,
            device_type=req.device_type,
            queue_position=len(self._pending_queue) + 1,
            created_at=now,
            accepted_at=None,
            completed_at=None,
            charging_duration_s=None,
            energy_delivered_wh=None,
        )

        self._requests[request_id] = detail
        self._pending_queue.append(request_id)

        logger.info("새 요청 추가: %s (대기 %d번째)", request_id, detail.queue_position)
        return detail

    def get_request(self, request_id: str) -> Optional[ChargeRequestDetail]:
        """요청 조회"""
        return self._requests.get(request_id)

    def get_next_pending(self) -> Optional[ChargeRequestDetail]:
        """다음 대기 요청 반환 (제거하지 않음)"""
        if not self._pending_queue:
            return None
        return self._requests.get(self._pending_queue[0])

    def accept_request(self, request_id: str) -> bool:
        """요청 수락 (PENDING → ACCEPTED)"""
        if request_id not in self._requests:
            return False

        detail = self._requests[request_id]
        if detail.status != RequestStatus.PENDING:
            return False

        detail.status = RequestStatus.ACCEPTED
        detail.accepted_at = datetime.now()

        if request_id in self._pending_queue:
            self._pending_queue.remove(request_id)

        # 대기열 순서 업데이트
        self._update_queue_positions()

        logger.info("요청 수락: %s", request_id)
        return True

    def start_request(self, request_id: str) -> bool:
        """요청 시작 (ACCEPTED → IN_PROGRESS)"""
        if request_id not in self._requests:
            return False

        detail = self._requests[request_id]
        if detail.status != RequestStatus.ACCEPTED:
            return False

        detail.status = RequestStatus.IN_PROGRESS
        return True

    def complete_request(
        self,
        request_id: str,
        charging_duration_s: float = 0,
        energy_delivered_wh: float = 0,
    ) -> bool:
        """요청 완료"""
        if request_id not in self._requests:
            return False

        detail = self._requests[request_id]
        detail.status = RequestStatus.COMPLETED
        detail.completed_at = datetime.now()
        detail.charging_duration_s = charging_duration_s
        detail.energy_delivered_wh = energy_delivered_wh

        self._completed_today += 1
        logger.info(
            "요청 완료: %s (%.1f초, %.1fWh)",
            request_id,
            charging_duration_s,
            energy_delivered_wh,
        )
        return True

    def cancel_request(self, request_id: str) -> bool:
        """요청 취소"""
        if request_id not in self._requests:
            return False

        detail = self._requests[request_id]
        if detail.status in (RequestStatus.COMPLETED, RequestStatus.CANCELLED):
            return False

        detail.status = RequestStatus.CANCELLED

        if request_id in self._pending_queue:
            self._pending_queue.remove(request_id)
            self._update_queue_positions()

        logger.info("요청 취소: %s", request_id)
        return True

    def get_queue_status(self) -> QueueStatus:
        """대기열 상태 조회"""
        # 최근 요청만 반환 (완료 포함 최대 20개)
        recent = sorted(
            self._requests.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )[:20]

        return QueueStatus(
            total_pending=len(self._pending_queue),
            total_completed_today=self._completed_today,
            requests=recent,
        )

    def _update_queue_positions(self) -> None:
        """대기열 순서 업데이트"""
        for i, req_id in enumerate(self._pending_queue):
            if req_id in self._requests:
                self._requests[req_id].queue_position = i + 1


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """WebSocket 연결 관리자"""

    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(
            "WebSocket 연결: %s (총 %d)", websocket.client, len(self._connections)
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info("WebSocket 해제: 총 %d", len(self._connections))

    async def broadcast(self, message: dict) -> None:
        """모든 클라이언트에 메시지 전송"""
        for connection in self._connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

    async def broadcast_robot_update(self, update: WSRobotUpdate) -> None:
        """로봇 상태 업데이트 브로드캐스트"""
        await self.broadcast(
            {
                "type": "robot_update",
                "data": update.model_dump(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def broadcast_request_update(self, update: WSRequestUpdate) -> None:
        """요청 상태 업데이트 브로드캐스트"""
        await self.broadcast(
            {
                "type": "request_update",
                "data": update.model_dump(),
                "timestamp": datetime.now().isoformat(),
            }
        )


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

queue_manager = RequestQueueManager()
connection_manager = ConnectionManager()

# 로봇 상태 (메인 루프에서 업데이트)
robot_state: Dict = {
    "status": RobotStatus.IDLE,
    "latitude": 37.5665,
    "longitude": 126.9780,
    "heading": 0.0,
    "speed_kmh": 0.0,
    "robot_soc": 80.0,
    "output_soc": 90.0,
    "is_robot_low": False,
    "is_output_charging": False,
    "output_current_ma": 0.0,
    "current_request_id": None,
    "state_duration_s": 0.0,
    "start_time": time.time(),
}


def update_robot_state(
    status: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    robot_soc: Optional[float] = None,
    output_soc: Optional[float] = None,
    current_request_id: Optional[str] = None,
    **kwargs,
) -> None:
    """로봇 상태 업데이트 (메인 루프에서 호출)"""
    if status:
        robot_state["status"] = status
    if latitude is not None:
        robot_state["latitude"] = latitude
    if longitude is not None:
        robot_state["longitude"] = longitude
    if robot_soc is not None:
        robot_state["robot_soc"] = robot_soc
    if output_soc is not None:
        robot_state["output_soc"] = output_soc
    if current_request_id is not None:
        robot_state["current_request_id"] = current_request_id

    for k, v in kwargs.items():
        if k in robot_state:
            robot_state[k] = v


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클"""
    logger.info("ChargeMate API 서버 시작")
    yield
    logger.info("ChargeMate API 서버 종료")


app = FastAPI(
    title="ChargeMate API",
    description="자율주행 이동형 충전 로봇 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
_PROJECT_ROOT = Path(__file__).parent.parent
_WEB_APP_DIR = _PROJECT_ROOT / "web_app"
_STATIC_DIR = _WEB_APP_DIR / "static"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    logger.info("정적 파일 디렉토리 마운트: %s", _STATIC_DIR)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """웹 앱 메인 페이지 (index.html)"""
    index_path = _WEB_APP_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "ChargeMate API v1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "uptime_s": time.time() - robot_state["start_time"]}


# --- 충전 요청 ---


@app.post("/api/request", response_model=ChargeRequestResponse)
async def create_charge_request(req: ChargeRequest):
    """충전 요청 생성"""
    detail = queue_manager.add_request(req)

    # 예상 대기 시간 계산 (간단히 1요청당 10분)
    estimated_wait = detail.queue_position * 10.0 if detail.queue_position else 0

    return ChargeRequestResponse(
        request_id=detail.request_id,
        status=detail.status,
        queue_position=detail.queue_position,
        estimated_wait_minutes=estimated_wait,
        created_at=detail.created_at,
    )


@app.get("/api/request/{request_id}", response_model=ChargeRequestDetail)
async def get_request(request_id: str):
    """요청 상세 조회"""
    detail = queue_manager.get_request(request_id)
    if not detail:
        raise HTTPException(status_code=404, detail="요청을 찾을 수 없습니다")
    return detail


@app.delete("/api/request/{request_id}")
async def cancel_request(request_id: str):
    """요청 취소"""
    success = queue_manager.cancel_request(request_id)
    if not success:
        raise HTTPException(status_code=400, detail="취소할 수 없는 요청입니다")

    # WebSocket 브로드캐스트
    await connection_manager.broadcast_request_update(
        WSRequestUpdate(
            request_id=request_id,
            status=RequestStatus.CANCELLED,
            queue_position=None,
            message="요청이 취소되었습니다",
        )
    )

    return {"success": True, "message": "요청이 취소되었습니다"}


# --- 대기열 ---


@app.get("/api/queue", response_model=QueueStatus)
async def get_queue():
    """대기열 상태 조회"""
    return queue_manager.get_queue_status()


# --- 로봇 상태 ---


@app.get("/api/robot/status", response_model=RobotStatusResponse)
async def get_robot_status():
    """로봇 상태 조회"""
    return RobotStatusResponse(
        status=robot_state["status"],
        position=RobotPosition(
            latitude=robot_state["latitude"],
            longitude=robot_state["longitude"],
            heading=robot_state.get("heading"),
            speed_kmh=robot_state.get("speed_kmh"),
        ),
        battery=BatteryInfo(
            robot_soc=robot_state["robot_soc"],
            output_soc=robot_state["output_soc"],
            is_robot_low=robot_state.get("is_robot_low", False),
            is_output_charging=robot_state.get("is_output_charging", False),
            output_current_ma=robot_state.get("output_current_ma"),
        ),
        current_request_id=robot_state.get("current_request_id"),
        state_duration_s=robot_state.get("state_duration_s", 0),
        uptime_s=time.time() - robot_state["start_time"],
    )


@app.post("/api/robot/emergency-stop")
async def emergency_stop():
    """비상 정지"""
    robot_state["status"] = RobotStatus.EMERGENCY_STOP
    logger.warning("비상 정지 명령 수신")
    return {"success": True, "message": "비상 정지 활성화"}


# --- WebSocket ---


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 상태 WebSocket"""
    await connection_manager.connect(websocket)

    try:
        while True:
            # 클라이언트 메시지 수신 (heartbeat 등)
            _ = await websocket.receive_text()

            # 현재 상태 전송
            await websocket.send_json(
                {
                    "type": "robot_update",
                    "data": {
                        "status": robot_state["status"],
                        "latitude": robot_state["latitude"],
                        "longitude": robot_state["longitude"],
                        "robot_soc": robot_state["robot_soc"],
                        "output_soc": robot_state["output_soc"],
                        "current_request_id": robot_state.get("current_request_id"),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Server Runner
# ---------------------------------------------------------------------------


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """서버 실행 (별도 스레드용)"""
    uvicorn.run(app, host=host, port=port, log_level="info")


async def run_server_async(host: str = "0.0.0.0", port: int = 8000) -> None:
    """서버 비동기 실행"""
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
