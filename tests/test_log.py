import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import StringIO
import importlib
import inspect
import json
import re
import sys

import pytest
from loguru import logger as host_logger
from rich.table import Table
from scipykit.log import (
    log, configure_log, get_logger, log_context, log_to, stop_log, record_to,
    PRESETS, THEMES,
)


@pytest.fixture(autouse=True)
def terminal():
    stream = StringIO()
    configure_log(stream=stream, colors=False)
    yield stream
    stop_log()
    configure_log()


def test_default_print_semantics_no_files(tmp_path, monkeypatch, terminal):
    monkeypatch.chdir(tmp_path)
    log('literal {x}', {'a': 1}, sep=' / ')
    log.debug('hidden')
    output = terminal.getvalue()
    assert 'literal {x} /' in output and "{'a': 1}" in output
    assert 'hidden' not in output
    assert not list(tmp_path.iterdir())


def test_one_line_context_persistent_and_opt_out(tmp_path):
    one, span, persistent = (tmp_path / name for name in ('one.log', 'span.log', 'all.log'))
    log_to(persistent)
    log('only-once', file=one)
    with record_to(span):
        log('scoped')
        log('terminal-only', file=False)
    log('after')
    stop_log()
    log('stopped')
    assert 'only-once' in one.read_text() and 'after' not in one.read_text()
    assert 'scoped' in span.read_text() and 'after' not in span.read_text()
    assert 'terminal-only' not in persistent.read_text()
    assert 'after' in persistent.read_text() and 'stopped' not in persistent.read_text()


def test_nested_same_path_is_not_duplicated_and_closes_after_error(tmp_path):
    path = tmp_path / 'nested.log'
    with pytest.raises(RuntimeError), record_to(path):
        with record_to(path):
            log('unique-message')
        raise RuntimeError('test')
    log('outside')
    assert path.read_text().count('unique-message') == 1
    assert 'outside' not in path.read_text()
    from scipykit.log._state import state
    assert not state.files


def test_caller_line_bound_context_and_json(tmp_path):
    path = tmp_path / 'records.jsonl'
    bound = get_logger('loader', run='exp1')
    with record_to(path, serialize=True), log_context(stage='read'):
        expected_line = inspect.currentframe().f_lineno + 1
        bound.warning('payload', epoch=4)
    record = json.loads(path.read_text())
    assert record['line'] == expected_line
    assert record['function'] == 'test_caller_line_bound_context_and_json'
    assert record['file'] == __file__
    assert record['extra']['component'] == 'loader'
    assert record['extra']['stage'] == 'read'
    assert record['extra']['epoch'] == 4
    assert not any(k.startswith('_') for k in record['extra'])


def test_depth_escape_hatch(tmp_path):
    path = tmp_path / 'depth.jsonl'
    def wrapper(message):
        log(message, depth=1)
    with record_to(path, serialize=True):
        expected_line = inspect.currentframe().f_lineno + 1
        wrapper('caller')
    record = json.loads(path.read_text())
    assert record['line'] == expected_line
    assert record['function'] == 'test_depth_escape_hatch'


def test_ansi_markup_custom_theme_rich_and_exception_clean_files(tmp_path):
    terminal = StringIO()
    configure_log(stream=terminal, colors=True, theme={'INFO': 'bold magenta'})
    plain, structured = tmp_path / 'plain.log', tmp_path / 'structured.jsonl'
    with record_to(plain), record_to(structured, serialize=True):
        log('\033[31mred\033[0m [green]green[/green]', markup=True,
            owner='\033[32m中文\033[0m')
        log('\033]8;;https://example.com\033\\link\033]8;;\033\\')
        table = Table('name', 'score')
        table.add_row('中文', '0.9')
        log(table)
        try:
            raise ValueError('\033[31mbad\033[0m')
        except ValueError:
            log.exception('failed')
    assert '\x1b[' in terminal.getvalue()
    for path in (plain, structured):
        output = path.read_text()
        assert '\x1b' not in output
        assert '[green]' not in output
        assert 'ValueError' in output and '中文' in output
    records = [json.loads(line) for line in structured.read_text().splitlines()]
    assert records[0]['message'] == 'red green'
    assert records[-1]['exception'].endswith('ValueError: bad\n')


def test_reconfigure_reload_and_host_loguru_isolation(tmp_path):
    host = StringIO()
    handler = host_logger.add(host, format='{message}')
    terminal = StringIO()
    try:
        configure_log(stream=terminal, colors=False)
        configure_log(stream=terminal, colors=False)
        importlib.reload(importlib.import_module('scipykit.log'))
        log('local-message')
        host_logger.info('host-message')
        assert terminal.getvalue().count('local-message') == 1
        assert 'host-message' not in terminal.getvalue()
        assert 'local-message' not in host.getvalue()
        assert 'host-message' in host.getvalue()
    finally:
        host_logger.remove(handler)


@pytest.mark.parametrize('preset', PRESETS)
@pytest.mark.parametrize('theme', THEMES)
def test_presets_and_themes(preset, theme):
    stream = StringIO()
    configure_log(preset=preset, theme=theme, stream=stream, colors=True)
    log.success('works')
    assert 'works' in stream.getvalue()


def test_custom_format_threshold_and_file_rotation(tmp_path):
    stream = StringIO()
    configure_log(format='{extra[scope]}{level.name}/{message}', level='WARNING', stream=stream)
    path = tmp_path / 'rotate.log'
    with record_to(path, rotation='500 B', retention=2, enqueue=True):
        for i in range(40):
            log.debug('rotation-data', i)
        log.warning('visible')
    assert 'rotation-data' not in stream.getvalue()
    assert 'WARNING/visible' in stream.getvalue()
    assert 1 < len(list(tmp_path.glob('*.log'))) <= 3
    assert 'visible' in path.read_text()


def test_async_contexts_do_not_leak(tmp_path):
    async def worker(name):
        with log_context(worker=name), record_to(tmp_path / f'{name}.jsonl', serialize=True):
            await asyncio.sleep(.01)
            log(name)
    async def run():
        await asyncio.gather(worker('first'), worker('second'))
    asyncio.run(run())
    for name in ('first', 'second'):
        records = [json.loads(line) for line in (tmp_path / f'{name}.jsonl').read_text().splitlines()]
        assert len(records) == 1
        assert records[0]['extra']['worker'] == name
        assert records[0]['message'] == name


def test_threads_share_file_without_duplicate_records(tmp_path):
    path = tmp_path / 'threads.jsonl'
    def worker(i):
        with record_to(path, serialize=True):
            log('thread', worker=i)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(12)))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(record['extra']['worker'] for record in records) == list(range(12))


def test_nonfinite_scientific_values_are_valid_json(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    with record_to(path, serialize=True):
        log('unstable', loss=float('nan'), bounds=[float('inf'), -float('inf')])
    record = json.loads(path.read_text())
    assert record['extra']['loss'] == 'nan'
    assert record['extra']['bounds'] == ['inf', '-inf']
