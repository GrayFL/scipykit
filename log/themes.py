"""布局模板与 Rich 颜色主题，可通过 configure_log 自定义。"""
PRESETS = {
    'minimal': '{level.name: <8} {message}',
    'compact': '{time:HH:mm:ss} | {level.name: <8} | {extra[scope]}{message}',
    'source': '{time:HH:mm:ss.SSS} | {level.name: <8} | {name}:{function}:{line} | {extra[scope]}{message}',
    'detailed': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level.name: <8} | {process.id}/{thread.id} | {file.path}:{line} ({function}) | {extra[scope]}{message}',
}
THEMES = {
    'default': {'TRACE': 'dim', 'DEBUG': 'cyan', 'INFO': 'blue', 'SUCCESS': 'bold green',
                'WARNING': 'bold yellow', 'ERROR': 'bold red', 'CRITICAL': 'bold white on red'},
    'pastel': {'TRACE': 'grey62', 'DEBUG': '#89b4fa', 'INFO': '#cdd6f4', 'SUCCESS': '#a6e3a1',
               'WARNING': '#f9e2af', 'ERROR': '#f38ba8', 'CRITICAL': 'bold #f5c2e7'},
    'light': {'TRACE': 'grey50', 'DEBUG': 'blue', 'INFO': 'black', 'SUCCESS': 'dark_green',
              'WARNING': 'dark_orange3', 'ERROR': 'red3', 'CRITICAL': 'bold dark_red'},
    'mono': {name: ('bold' if name in ('WARNING', 'ERROR', 'CRITICAL') else '')
             for name in ('TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL')},
}
