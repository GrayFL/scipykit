"""持有可重配置的终端 sink 和按路径引用计数的文件 sink。"""
from contextvars import ContextVar
from pathlib import Path
from threading import RLock
import sys
from rich.console import Console
from rich.style import Style
from rich.text import Text
from ._backend import new_backend
from .formatting import file_formatter, exception_text, strip_ansi
from .themes import PRESETS, THEMES

context = ContextVar('scipykit_log_context', default=None)
destinations = ContextVar('scipykit_log_destinations', default=())


class State:
    def __init__(self):
        self.backend = new_backend()
        self.lock = RLock()
        self.terminal_id = None
        self.files = {}
        self.persistent = set()
        self.initialized = False
        self.width = 100

    def configure(self, *, level='INFO', preset='source', theme='default',
                  format=None, colors=None, stream=None, width=100, terminal=True):
        template = PRESETS[preset] if format is None else format
        styles = THEMES[theme] if isinstance(theme, str) else {**THEMES['default'], **theme}
        styles = {key.upper(): Style.parse(value) for key, value in styles.items()}
        if width < 20:
            raise ValueError('width 应不小于 20')
        console = Console(file=stream if stream is not None else sys.stderr,
                          force_terminal=colors,
                          color_system='truecolor' if colors is True else ('auto' if colors is None else None),
                          no_color=False if colors is True else None,
                          force_jupyter=False, width=width, highlight=False)

        def sink(message):
            record = message.record
            rich_message = record['extra']['_display']
            before, marker, after = template.partition('{message}')
            if marker:
                output = Text(strip_ansi(before.format_map(record)))
                output.append_text(rich_message)
                output.append(strip_ansi(after.format_map(record)))
            else:
                output = Text(strip_ansi(template.format_map(record)))
            output.stylize_before(styles.get(record['level'].name, Style()))
            console.print(output, soft_wrap=True)
            error = exception_text(record)
            if error:
                console.print(Text(error.rstrip(), style=styles.get('ERROR', Style())), soft_wrap=True)

        with self.lock:
            # 先验证并创建新 sink，再撤掉旧的；无效配置不会破坏当前输出。
            handler = self.backend.add(sink, level=level, format='{message}',
                                       colorize=False, catch=False, diagnose=False)
            if self.terminal_id is not None:
                self.backend.remove(self.terminal_id)
            self.terminal_id = handler if terminal else None
            if not terminal:
                self.backend.remove(handler)
            self.width = width
            self.initialized = True

    def ensure(self):
        with self.lock:
            if not self.initialized:
                self.configure()

    def acquire(self, path, **options):
        key = str(Path(path).expanduser().resolve())
        with self.lock:
            if key in self.files:
                entry = self.files[key]
                if options and options != entry['options']:
                    raise ValueError(f'此路径已打开且配置不同，请先停止记录: {key}')
                entry['refs'] += 1
            else:
                settings = dict(options)
                serialize = settings.pop('serialize', False)
                template = settings.pop('format', PRESETS['source'])
                level = settings.pop('level', 'TRACE')
                # 文件输出始终清洁；禁止用原生 serialize 包含内部 Rich 对象。
                forbidden = set(settings) & {'colorize', 'filter', 'catch', 'diagnose', 'backtrace', 'encoding'}
                if forbidden:
                    raise ValueError(f'由封装管理的文件参数: {sorted(forbidden)}')
                Path(key).parent.mkdir(parents=True, exist_ok=True)
                handler = self.backend.add(key, level=level, colorize=False, catch=False,
                    diagnose=False, backtrace=False, encoding='utf-8',
                    format=file_formatter(template, serialize),
                    filter=lambda record: key in record['extra']['_files'], **settings)
                self.files[key] = {'id': handler, 'refs': 1, 'options': options}
        return key

    def release(self, key):
        with self.lock:
            entry = self.files[key]
            entry['refs'] -= 1
            if entry['refs'] == 0:
                self.backend.remove(entry['id'])
                del self.files[key]


state = State()
