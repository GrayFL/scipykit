"""Loguru 隔离适配层，唯一依赖其内部 API 的位置。

Loguru 0.7.x 没有公开的独立 logger 工厂；全局实例携带不可 deepcopy
的终端/文件 sink。单独创建 Core 避免删除或接管宿主项目的 handlers。
升级 Loguru 时需运行 tests/test_log.py 的隔离与溯源测试。
"""
from loguru._logger import Core, Logger


def new_backend():
    return Logger(core=Core(), exception=None, depth=0, record=False,
                  lazy=False, colors=False, raw=False, capture=True,
                  patchers=[], extra={})
