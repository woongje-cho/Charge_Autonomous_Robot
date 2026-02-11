"""
Bridge Guardian — TTS (Text-to-Speech) 서비스
한국어 음성 메시지를 통해 위기 상황에서 대상자에게 안내를 제공한다.

엔진 우선순위:
  1. pyttsx3 (오프라인, Jetson 최적)
  2. gTTS   (온라인 대체)
  3. espeak  (최후 수단, subprocess)

절대 크래시하지 않는다 — 모든 예외를 내부에서 처리한다.
"""

import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 선택적 의존성
# ---------------------------------------------------------------------------
try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS as _gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# 메시지 타입 열거형
# ---------------------------------------------------------------------------
class MessageType(str, Enum):
    GREETING = "greeting"
    HELP_OFFER = "help_offer"
    NOT_ALONE = "not_alone"
    HELP_COMING = "help_coming"
    EMERGENCY = "emergency"
    CALM = "calm"
    PATROL_ANNOUNCE = "patrol_announce"


# ---------------------------------------------------------------------------
# 한국어 기본 메시지 사전
# ---------------------------------------------------------------------------
MESSAGES: Dict[str, str] = {
    MessageType.GREETING:        "안녕하세요, 괜찮으신가요?",
    MessageType.HELP_OFFER:      "도움이 필요하시면 말씀해주세요. 저는 여러분을 돕기 위해 여기 있습니다.",
    MessageType.NOT_ALONE:       "혼자가 아닙니다. 자살예방상담전화 1393으로 연락하실 수 있습니다.",
    MessageType.HELP_COMING:     "잠시만 기다려주세요, 도움을 요청하고 있습니다.",
    MessageType.EMERGENCY:       "긴급 상황입니다. 119에 연락하고 있습니다. 안전한 곳으로 이동해주세요.",
    MessageType.CALM:            "천천히 심호흡 해보세요. 괜찮습니다.",
    MessageType.PATROL_ANNOUNCE: "안전 순찰 중입니다. 도움이 필요하시면 말씀해주세요.",
}


# ---------------------------------------------------------------------------
# TTSMessage 데이터 클래스
# ---------------------------------------------------------------------------
@dataclass
class TTSMessage:
    """재생 대기열에 넣을 TTS 메시지."""
    text: str
    priority: int = 5          # 낮을수록 우선 (1=최우선, 10=최저)
    repeat: int = 1            # 반복 횟수
    delay_between: float = 1.0 # 반복 사이 대기(초)
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other: "TTSMessage") -> bool:
        """PriorityQueue 비교용."""
        return self.priority < other.priority


# ---------------------------------------------------------------------------
# TTS 엔진 열거형
# ---------------------------------------------------------------------------
class _Engine(str, Enum):
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    ESPEAK = "espeak"
    NONE = "none"


# ---------------------------------------------------------------------------
# TTSService
# ---------------------------------------------------------------------------
class TTSService:
    """백그라운드 스레드에서 TTS 메시지를 재생하는 서비스.

    Parameters
    ----------
    config : dict, optional
        settings.yaml ``tts`` 섹션. 키:
        - enabled (bool, default True)
        - engine (str, default "pyttsx3")
        - language (str, default "ko")
        - rate (int, default 150) — pyttsx3 말하기 속도
        - volume (float, default 0.9) — 0.0~1.0
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self._enabled: bool = cfg.get("enabled", True)
        self._language: str = cfg.get("language", "ko")
        self._rate: int = int(cfg.get("rate", 150))
        self._volume: float = float(cfg.get("volume", 0.9))

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._engine_type: _Engine = _Engine.NONE
        self._pyttsx3_engine = None

        # 엔진 초기화
        preferred = cfg.get("engine", "pyttsx3").lower()
        self._init_engine(preferred)

        # 워커 스레드 시작
        self._thread = threading.Thread(
            target=self._worker, name="tts-worker", daemon=True
        )
        self._thread.start()
        logger.info(
            "TTSService 초기화 완료 (엔진=%s, 언어=%s)",
            self._engine_type.value, self._language,
        )

    # ------------------------------------------------------------------
    # 엔진 초기화
    # ------------------------------------------------------------------

    def _init_engine(self, preferred: str) -> None:
        """우선순위: pyttsx3 → gTTS → espeak → none."""
        # 1) pyttsx3
        if preferred == "pyttsx3" or preferred == "auto":
            if _PYTTSX3_AVAILABLE:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self._rate)
                    engine.setProperty("volume", self._volume)
                    # 한국어 음성 탐색
                    voices = engine.getProperty("voices")
                    for v in voices:
                        if "ko" in (v.id or "").lower() or "korean" in (v.name or "").lower():
                            engine.setProperty("voice", v.id)
                            break
                    self._pyttsx3_engine = engine
                    self._engine_type = _Engine.PYTTSX3
                    return
                except Exception as exc:
                    logger.warning("pyttsx3 초기화 실패: %s — 대체 엔진 시도", exc)

        # 2) gTTS
        if _GTTS_AVAILABLE:
            self._engine_type = _Engine.GTTS
            return

        # 3) espeak (subprocess)
        try:
            subprocess.run(
                ["espeak", "--version"],
                capture_output=True, timeout=5,
            )
            self._engine_type = _Engine.ESPEAK
            return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        logger.error("사용 가능한 TTS 엔진이 없습니다. TTS 비활성화됨.")
        self._engine_type = _Engine.NONE
        self._enabled = False

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def speak(self, message_type: str, custom_text: Optional[str] = None,
              priority: int = 5, repeat: int = 1) -> None:
        """메시지를 재생 큐에 추가한다.

        Parameters
        ----------
        message_type : str
            MESSAGES 딕셔너리 키 또는 MessageType 값.
        custom_text : str, optional
            사전 정의 메시지 대신 사용할 커스텀 텍스트.
        priority : int
            1(최우선) ~ 10(최저). 기본 5.
        repeat : int
            반복 횟수.
        """
        if not self._enabled:
            return

        text = custom_text or MESSAGES.get(message_type, "")
        if not text:
            logger.warning("알 수 없는 메시지 타입: %s", message_type)
            return

        msg = TTSMessage(text=text, priority=priority, repeat=repeat)
        self._queue.put(msg)
        logger.debug("TTS 큐 추가: [P%d] %s", priority, text[:30])

    def speak_text(self, text: str, priority: int = 5, repeat: int = 1) -> None:
        """임의의 텍스트를 재생 큐에 추가한다."""
        if not self._enabled or not text:
            return
        msg = TTSMessage(text=text, priority=priority, repeat=repeat)
        self._queue.put(msg)

    def stop(self) -> None:
        """워커 스레드 종료를 요청한다."""
        self._stop_event.set()
        # 큐에 센티널 추가해 블로킹 해제
        self._queue.put(TTSMessage(text="", priority=99))
        logger.info("TTSService 종료 요청")

    @property
    def engine_name(self) -> str:
        return self._engine_type.value

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # 백그라운드 워커
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """큐에서 메시지를 꺼내 순차적으로 재생한다."""
        while not self._stop_event.is_set():
            try:
                msg: TTSMessage = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._stop_event.is_set() or not msg.text:
                break

            for i in range(msg.repeat):
                if self._stop_event.is_set():
                    break
                try:
                    self._play(msg.text)
                except Exception as exc:
                    logger.error("TTS 재생 실패: %s", exc)
                if i < msg.repeat - 1:
                    time.sleep(msg.delay_between)

            self._queue.task_done()

        logger.info("TTS 워커 스레드 종료")

    # ------------------------------------------------------------------
    # 엔진별 재생
    # ------------------------------------------------------------------

    def _play(self, text: str) -> None:
        """현재 엔진으로 텍스트를 음성 재생한다."""
        if self._engine_type == _Engine.PYTTSX3:
            self._play_pyttsx3(text)
        elif self._engine_type == _Engine.GTTS:
            self._play_gtts(text)
        elif self._engine_type == _Engine.ESPEAK:
            self._play_espeak(text)

    def _play_pyttsx3(self, text: str) -> None:
        """pyttsx3 엔진으로 재생 (오프라인)."""
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as exc:
            logger.warning("pyttsx3 재생 실패, gTTS 대체 시도: %s", exc)
            if _GTTS_AVAILABLE:
                self._play_gtts(text)
            else:
                self._play_espeak(text)

    def _play_gtts(self, text: str) -> None:
        """gTTS + 임시 파일 → 시스템 오디오 재생."""
        tmp_path = None
        try:
            tts = _gTTS(text=text, lang=self._language)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                tmp_path = fp.name
                tts.save(tmp_path)

            # 플랫폼별 재생
            import platform
            system = platform.system()
            if system == "Linux":
                # Jetson / Linux
                subprocess.run(
                    ["mpg123", "-q", tmp_path],
                    capture_output=True, timeout=30,
                )
            elif system == "Darwin":
                subprocess.run(
                    ["afplay", tmp_path],
                    capture_output=True, timeout=30,
                )
            elif system == "Windows":
                # PowerShell을 통한 재생
                subprocess.run(
                    ["powershell", "-c",
                     f'(New-Object Media.SoundPlayer "{tmp_path}").PlaySync()'],
                    capture_output=True, timeout=30,
                )
        except Exception as exc:
            logger.warning("gTTS 재생 실패, espeak 대체 시도: %s", exc)
            self._play_espeak(text)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _play_espeak(self, text: str) -> None:
        """espeak를 subprocess로 실행 (최후 수단)."""
        try:
            subprocess.run(
                ["espeak", "-v", self._language, text],
                capture_output=True, timeout=30,
            )
        except Exception as exc:
            logger.error("espeak 재생 실패 (모든 엔진 실패): %s", exc)
