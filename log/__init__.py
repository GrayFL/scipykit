"""Loguru + Rich 的随手日志入口；默认只在首次 log 时启用终端输出。"""
from .core import (Log, log, lprint, configure_log, get_logger,
                   log_context, log_to, stop_log, record_to)
from .themes import PRESETS, THEMES
from .formatting import strip_ansi

__all__ = ['Log', 'log', 'lprint', 'configure_log', 'get_logger', 'log_context',
           'log_to', 'stop_log', 'record_to', 'PRESETS', 'THEMES', 'strip_ansi']
