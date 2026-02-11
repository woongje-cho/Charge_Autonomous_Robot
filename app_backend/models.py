"""
ChargeMate - API 데이터 모델
Pydantic 모델 정의.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RequestStatus(str, Enum):
    """충전 요청 상태"""

    PENDING = "pending"  # 대기 중
    ACCEPTED = "accepted"  # 수락됨 (로봇 배차)
    IN_PROGRESS = "in_progress"  # 진행 중 (이동/충전)
    COMPLETED = "completed"  # 완료
    CANCELLED = "cancelled"  # 취소됨
    FAILED = "failed"  # 실패


class RobotStatus(str, Enum):
    """로봇 상태"""

    IDLE = "idle"
    DISPATCHING = "dispatching"
    NAVIGATING = "navigating"
    APPROACHING_USER = "approaching_user"
    DOCKING = "docking"
    CHARGING = "charging"
    UNDOCKING = "undocking"
    RETURNING = "returning"
    EMERGENCY_STOP = "emergency_stop"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ChargeRequest(BaseModel):
    """충전 요청 생성"""

    latitude: float = Field(..., ge=-90, le=90, description="위도")
    longitude: float = Field(..., ge=-180, le=180, description="경도")
    user_name: Optional[str] = Field(None, max_length=50, description="사용자 이름")
    device_type: Optional[str] = Field(None, description="기기 유형 (phone, laptop 등)")

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 37.5665,
                "longitude": 126.9780,
                "user_name": "홍길동",
                "device_type": "laptop",
            }
        }


class ChargeRequestResponse(BaseModel):
    """충전 요청 응답"""

    request_id: str = Field(..., description="요청 ID")
    status: RequestStatus = Field(..., description="요청 상태")
    queue_position: Optional[int] = Field(None, description="대기열 순서")
    estimated_wait_minutes: Optional[float] = Field(
        None, description="예상 대기 시간 (분)"
    )
    created_at: datetime = Field(..., description="생성 시간")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_abc123",
                "status": "pending",
                "queue_position": 1,
                "estimated_wait_minutes": 5.0,
                "created_at": "2024-01-15T10:30:00",
            }
        }


class ChargeRequestDetail(BaseModel):
    """충전 요청 상세 정보"""

    request_id: str
    status: RequestStatus
    latitude: float
    longitude: float
    user_name: Optional[str]
    device_type: Optional[str]
    queue_position: Optional[int]
    created_at: datetime
    accepted_at: Optional[datetime]
    completed_at: Optional[datetime]
    charging_duration_s: Optional[float]
    energy_delivered_wh: Optional[float]


# ---------------------------------------------------------------------------
# Robot Status Models
# ---------------------------------------------------------------------------


class RobotPosition(BaseModel):
    """로봇 위치"""

    latitude: float
    longitude: float
    heading: Optional[float] = Field(None, description="방향 (도)")
    speed_kmh: Optional[float] = Field(None, description="속도 (km/h)")


class BatteryInfo(BaseModel):
    """배터리 정보"""

    robot_soc: float = Field(..., ge=0, le=100, description="로봇 배터리 (%)")
    output_soc: float = Field(..., ge=0, le=100, description="출력 배터리 (%)")
    is_robot_low: bool = Field(..., description="로봇 배터리 부족")
    is_output_charging: bool = Field(..., description="기기 충전 중")
    output_current_ma: Optional[float] = Field(None, description="출력 전류 (mA)")


class RobotStatusResponse(BaseModel):
    """로봇 상태 응답"""

    status: RobotStatus
    position: Optional[RobotPosition]
    battery: BatteryInfo
    current_request_id: Optional[str]
    state_duration_s: float
    uptime_s: float

    class Config:
        json_schema_extra = {
            "example": {
                "status": "idle",
                "position": {
                    "latitude": 37.5665,
                    "longitude": 126.9780,
                    "heading": 45.0,
                    "speed_kmh": 0.0,
                },
                "battery": {
                    "robot_soc": 85.0,
                    "output_soc": 92.0,
                    "is_robot_low": False,
                    "is_output_charging": False,
                    "output_current_ma": 0.0,
                },
                "current_request_id": None,
                "state_duration_s": 120.5,
                "uptime_s": 3600.0,
            }
        }


# ---------------------------------------------------------------------------
# Queue Models
# ---------------------------------------------------------------------------


class QueueStatus(BaseModel):
    """대기열 상태"""

    total_pending: int = Field(..., description="대기 중인 요청 수")
    total_completed_today: int = Field(..., description="오늘 완료된 요청 수")
    requests: list[ChargeRequestDetail] = Field(..., description="요청 목록")


# ---------------------------------------------------------------------------
# WebSocket Models
# ---------------------------------------------------------------------------


class WSMessage(BaseModel):
    """WebSocket 메시지"""

    type: str = Field(..., description="메시지 유형")
    data: dict = Field(..., description="메시지 데이터")
    timestamp: datetime = Field(default_factory=datetime.now)


class WSRobotUpdate(BaseModel):
    """WebSocket 로봇 상태 업데이트"""

    status: RobotStatus
    latitude: float
    longitude: float
    robot_soc: float
    output_soc: float
    current_request_id: Optional[str]


class WSRequestUpdate(BaseModel):
    """WebSocket 요청 상태 업데이트"""

    request_id: str
    status: RequestStatus
    queue_position: Optional[int]
    message: Optional[str]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def generate_request_id() -> str:
    """고유 요청 ID 생성"""
    return f"req_{uuid.uuid4().hex[:12]}"
