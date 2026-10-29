"""
================================================================================
工具辅助模块
================================================================================

提供日志配置、时间统计、打印格式化等通用工具函数。
"""

import logging
import time
from pathlib import Path
from config.settings import LOG_CONFIG, LOG_DIR


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    配置全局日志系统。

    Args:
        level:    日志级别 (DEBUG / INFO / WARNING / ERROR)，默认 INFO
        log_file: 日志文件路径，默认自动生成到 logs/ 目录
    """
    log_level = getattr(logging, (level or LOG_CONFIG["level"]).upper(), logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"], LOG_CONFIG["datefmt"]))

    # 文件输出
    if log_file is None:
        log_file = str(LOG_DIR / f"rag_{time.strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"], LOG_CONFIG["datefmt"]))

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info("日志系统已配置: 控制台级别=%s, 文件路径=%s", level or LOG_CONFIG["level"], log_file)


class Timer:
    """简单的计时器上下文管理器"""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            logging.info("[计时] %s: %.2f 秒", self.name, self.elapsed)


def print_banner(title: str, width: int = 70) -> None:
    """打印格式化的横幅标题。"""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num: int, title: str, description: str = "") -> None:
    """打印步骤标题。"""
    print(f"\n{'─' * 60}")
    print(f"  [步骤 {step_num}] {title}")
    if description:
        print(f"  → {description}")
    print(f"{'─' * 60}")
