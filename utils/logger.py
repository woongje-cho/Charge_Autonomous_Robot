"""
Bridge Guardian — 로깅 유틸리티
"""

import logging
import sys
from pathlib import Path

# ANSI 색상 코드
_COLORS = {
    "DEBUG":    "\033[90m",    # 회색
    "INFO":     "\033[92m",    # 녹색
    "WARNING":  "\033[93m",    # 노란색
    "ERROR":    "\033[91m",    # 빨간색
    "CRITICAL": "\033[1;91m",  # 굵은 빨간색
}
_RESET = "\033[0m"


class _ColoredFormatter(logging.Formatter):
    """레벨별 색상이 적용되는 콘솔 포맷터"""

    def __init__(self, fmt: str, datefmt: str = ""):
        super().__init__(fmt, datefmt if datefmt else None)

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        record.levelname_colored = f"{color}{record.levelname:<8}{_RESET}"
        try:
            return super().format(record)
        except (KeyError, ValueError):
            # Fallback if format string references unavailable attributes
            return f"[{record.levelname}] [{record.name}] {record.getMessage()}"


def setup_logging(level: str = "INFO", log_file: str = "") -> None:
    """전역 로깅 설정"""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 기존 핸들러 제거
    root.handlers.clear()

    # 콘솔 핸들러
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s] %(levelname_colored)s [%(name)s] %(message)s"
    console.setFormatter(_ColoredFormatter(fmt, "%Y-%m-%d %H:%M:%S"))
    root.addHandler(console)

    # 파일 핸들러 (선택)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(path), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        plain_fmt = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
        fh.setFormatter(logging.Formatter(plain_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 반환"""
    return logging.getLogger(name)
