"""像带级别的 print 一样使用，并提供手动业务上下文与三种落盘粒度。"""
from contextlib import contextmanager
from pathlib import Path
from ._state import state, context, destinations
from .formatting import render_objects, clean_value


def configure_log(**kwargs):
    """配置终端：level/preset/theme/format/colors/stream/width/terminal。

    重复调用替换终端配置，已有文件输出保持。默认 INFO/source/default。
    theme 可传 {级别: Rich 样式} 字典；format 使用 Loguru record 占位符。
    """
    state.configure(**kwargs)


def log_to(path, **options):
    """持续落盘直到 stop_log；支持 rotation/retention/compression/enqueue 等。"""
    state.ensure()
    key = str(Path(path).expanduser().resolve())
    with state.lock:
        if key not in state.persistent:
            state.acquire(key, **options)
            state.persistent.add(key)
        elif options and options != state.files[key]['options']:
            raise ValueError('此路径已启用；修改配置前请调用 stop_log(path)')
    return Path(key)


def stop_log(path=None):
    """停止一个/全部持续文件输出；不影响终端和仍在运行的 record_to。"""
    with state.lock:
        keys = list(state.persistent) if path is None else [str(Path(path).expanduser().resolve())]
        for key in keys:
            if key in state.persistent:
                state.persistent.remove(key)
                state.release(key)


@contextmanager
def record_to(path, **options):
    """只记录当前线程/asyncio 上下文内的语句；支持嵌套，异常时也关闭。"""
    state.ensure()
    key = state.acquire(path, **options)
    token = destinations.set((*destinations.get(), key))
    try:
        yield Path(key)
    finally:
        destinations.reset(token)
        state.release(key)


@contextmanager
def log_context(**fields):
    """手动给一段代码绑定 run/component/stage 等信息，不扫描整个工程。"""
    token = context.set({**(context.get() or {}), **fields})
    try:
        yield
    finally:
        context.reset(token)


class Log:
    """可调用日志对象。所有位置参数按 sep 拼接，花括号不会隐式格式化。"""
    def __init__(self, **fields):
        self._fields = fields

    def bind(self, **fields):
        return type(self)(**{**self._fields, **fields})

    def _emit(self, *objects, level='INFO', sep=' ', file=None, pretty=True,
              markup=False, style=None, depth=0, exception=None, **fields):
        if not isinstance(depth, int) or depth < 0:
            raise ValueError('depth 必须为非负整数')
        state.ensure()
        text = render_objects(objects, sep=sep, pretty=pretty, markup=markup, width=state.width)
        if style:
            text.stylize(style)
        extra = clean_value({**(context.get() or {}), **self._fields, **fields})
        if any(key.startswith('_') for key in extra):
            raise ValueError('以下划线开头的上下文字段保留给内部使用')
        extra['scope'] = ' '.join(f'{key}={value}' for key, value in extra.items())
        if extra['scope']:
            extra['scope'] = '[' + extra['scope'] + '] '
        acquired = None
        try:
            if file is not None and file is not False:
                if file is True:
                    raise TypeError('file 请传具体路径；file=False 表示本句仅输出终端')
                acquired = state.acquire(file)
            # 同步发射期间不移除文件 sink；异步队列由 Loguru remove() 排空。
            with state.lock:
                routes = set() if file is False else state.persistent | set(destinations.get())
                if acquired:
                    routes.add(acquired)
                state.backend.bind(**extra, _display=text, _files=tuple(routes)).opt(
                    depth=2 + depth, exception=exception).log(level.upper() if isinstance(level, str) else level, text.plain)
        finally:
            if acquired:
                state.release(acquired)

    def __call__(self, *objects, **kwargs):
        return self._emit(*objects, **kwargs)

    def trace(self, *objects, **kwargs):
        return self._emit(*objects, level='TRACE', **kwargs)

    def debug(self, *objects, **kwargs):
        return self._emit(*objects, level='DEBUG', **kwargs)

    def info(self, *objects, **kwargs):
        return self._emit(*objects, level='INFO', **kwargs)

    def success(self, *objects, **kwargs):
        return self._emit(*objects, level='SUCCESS', **kwargs)

    def warning(self, *objects, **kwargs):
        return self._emit(*objects, level='WARNING', **kwargs)

    def error(self, *objects, **kwargs):
        return self._emit(*objects, level='ERROR', **kwargs)

    def critical(self, *objects, **kwargs):
        return self._emit(*objects, level='CRITICAL', **kwargs)

    def exception(self, *objects, **kwargs):
        return self._emit(*objects, level='ERROR', exception=True, **kwargs)


log = Log()
lprint = log


def get_logger(component=None, **fields):
    """例如 get_logger('dataset', run='exp01')；不改变真实代码调用位置。"""
    return log.bind(**({'component': component} if component is not None else {}), **fields)
