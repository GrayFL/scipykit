"""把漂亮终端输出与可检索的纯文本/JSON 文件表示分开。"""
import io
import json
import math
import re
import traceback
from collections.abc import Mapping
from rich.console import Console
from rich.pretty import pretty_repr
from rich.text import Text

# CSI（颜色/光标）、OSC（超链接/标题）、单字符 ESC 以及非文本控制符。
_ANSI = re.compile(r'(?:\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]|\x9b[0-?]*[ -/]*[@-~])')
_CONTROL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')


def strip_ansi(value):
    """清理真实终端控制序列，保留换行和制表符。"""
    return _CONTROL.sub('', _ANSI.sub('', str(value)))


def clean_value(value):
    if isinstance(value, str):
        return strip_ansi(value)
    if isinstance(value, Mapping):
        return {strip_ansi(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return strip_ansi(str(value))


def render_objects(objects, *, sep=' ', pretty=True, markup=False, width=100):
    """保留 Rich Text 的样式；容器美化、Table/Panel 等转为终端文本。"""
    result = Text()
    for i, obj in enumerate(objects):
        if i:
            result.append(strip_ansi(sep))
        if isinstance(obj, Text):
            item = obj.copy()
            # 正常 Text 保留样式；带原始控制符时退化为清洁文本。
            if strip_ansi(item.plain) != item.plain:
                item = Text(strip_ansi(item.plain))
        elif isinstance(obj, str):
            cleaned = strip_ansi(obj)
            item = Text.from_markup(cleaned) if markup else Text(cleaned)
        elif hasattr(obj, '__rich_console__') or hasattr(obj, '__rich__'):
            stream = io.StringIO()
            Console(file=stream, width=width, color_system=None, force_terminal=False,
                    force_jupyter=False).print(obj, end='')
            item = Text(strip_ansi(stream.getvalue().rstrip('\n')))
        else:
            value = clean_value(obj) if isinstance(obj, (dict, list, tuple)) else obj
            item = Text(strip_ansi(pretty_repr(value, max_width=width) if pretty else str(value)))
        result.append_text(item)
    return result


def exception_text(record):
    error = record['exception']
    if error is None:
        return ''
    return strip_ansi(''.join(traceback.format_exception(error.type, error.value, error.traceback)))


def file_formatter(template, serialize=False):
    """使用 Loguru 原生文件 sink 的轮转/保留；格式化前清理所有输出。"""
    def formatter(record):
        extra = {k: clean_value(v) for k, v in record['extra'].items() if not k.startswith('_')}
        error = exception_text(record)
        if serialize:
            payload = {
                'time': record['time'].isoformat(), 'level': record['level'].name,
                'message': strip_ansi(record['message']), 'module': record['name'],
                'file': record['file'].path, 'function': record['function'], 'line': record['line'],
                'process': record['process'].id, 'thread': record['thread'].id,
                'extra': extra, 'exception': error or None,
            }
            rendered = json.dumps(clean_value(payload), ensure_ascii=False, allow_nan=False)
        else:
            values = {**record, 'extra': extra, 'exception': error}
            rendered = template.format_map(values)
            if error and '{exception}' not in template:
                rendered += '\n' + error
        record['extra']['_file_text'] = strip_ansi(rendered).rstrip('\n')
        return '{extra[_file_text]}\n'
    return formatter
