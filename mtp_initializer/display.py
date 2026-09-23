"""Notebook 显示与外置资源；保留 notebook-assets MIME 并提供标准回退。"""
import os
from pathlib import Path
from html import escape
from IPython.display import display, display_markdown, clear_output as clr, Image as IPyImage
from matplotlib.figure import Figure
from .figure import savefig

IMAGE_MIME = 'application/vnd.notebook-assets.image+json'
DF_MIME = 'application/vnd.notebook-assets.dataframe+json'


def inspect_notebook_name():
    """可用 NOTEBOOK_NAME 手动指定，避免依赖 notebook 前端推断。"""
    name = os.environ.get('NOTEBOOK_NAME'
                         ) or os.environ.get('__notebook_filename__')
    if name:
        return Path(name).name
    from IPython import get_ipython
    shell = get_ipython()
    if shell and shell.user_ns.get('__vsc_ipynb_file__'):
        return Path(shell.user_ns['__vsc_ipynb_file__']).name
    return 'notebook'


def _asset_path(key, suffix: str, assets_root, notebook_name):
    key = Path(str(key))
    if key.is_absolute() or '..' in key.parts or str(key) == '.':
        raise ValueError('key 必须是资源目录内的相对名称')
    root = Path(
        assets_root or os.environ.get('NOTEBOOK_ASSETS_ROOT', 'assets')
        )
    folder = Path(notebook_name or inspect_notebook_name()).name
    path = root.expanduser() / folder / (str(key) + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def disp(
        obj,
        key=None,
        df_formats=None,
        df_render_html=False,
        df_link_html=True,
        *,
        assets_root=None,
        notebook_name=None,
        embed=False,
        **kwargs
    ):
    """随手显示对象；key 非空时写入 assets/<notebook>/<key>.*。

    默认使用外置图像及标准 HTML 引用；embed=True 内嵌 PNG 以便分享。
    DataFrame 默认纯文本，命名后保存数据及完整外置 HTML。
    df_formats=None 自动选择可用的 Parquet 引擎，否则用 CSV。
    私有 MIME 中图像、数据及 HTML 路径均为绝对路径，返回值亦如此。
    显示参数透传 IPython.display.display，返回保存路径/路径字典。
    """
    if isinstance(obj, Figure):
        if key is None:
            display(obj, **kwargs)
            return None
        path = _asset_path(key, '.png', assets_root, notebook_name)
        savefig(obj, path, is_frame_on=True, alpha=obj.get_facecolor()[3])
        obj._scipykit_path = path
        data = {
            IMAGE_MIME: {
                'kind': 'image', 'image': str(path)
                },
            'text/plain': str(path)
            }
        if embed:
            data['image/png'] = IPyImage(filename=str(path))._repr_png_()
        else:
            href = escape(path.as_posix(), quote=True)
            data['text/html'
                ] = f'<img src="{href}?mtime={path.stat().st_mtime_ns}" alt="{escape(str(key))}">'
        display(data, raw=True, **kwargs)
        return path
    # 数组、字符串等不触发 pandas 导入。
    import sys
    pd = sys.modules.get('pandas')
    if pd is None or not isinstance(obj, (pd.DataFrame, pd.Series)):
        display(obj, **kwargs)
        return None
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
    with pd.option_context('display.expand_frame_repr', False):
        data = {'text/plain': repr(frame)}
    if key is None:
        if df_render_html:
            data['text/html'] = frame.to_html()
        display(data, raw=True, **kwargs)
        return None
    if df_formats is None:
        from importlib.util import find_spec
        df_formats = ('parquet', ) if any(
            find_spec(name) is not None
            for name in ('pyarrow', 'fastparquet')
            ) else ('csv', )
    formats = (df_formats, ) if isinstance(df_formats,
                                            str) else tuple(df_formats)
    if set(formats) - {'parquet', 'csv', 'xlsx'}:
        raise ValueError(f'不支持的 DataFrame 格式: {formats}')
    paths = {}
    for fmt in formats:
        path = _asset_path(key, '.' + fmt, assets_root, notebook_name)
        if fmt == 'parquet':
            frame.to_parquet(path)
        elif fmt == 'csv':
            frame.to_csv(path)
        else:
            frame.to_excel(path, index=False)
        paths[fmt] = path
    descriptor = {
        'kind': 'series' if isinstance(obj, pd.Series) else 'dataframe',
        'shape': ' x '.join(map(str, frame.shape)),
        'data': {
            fmt: str(path)
            for fmt, path in paths.items()
            }
        }
    if df_link_html:
        path = _asset_path(key, '.html', assets_root, notebook_name)
        path.write_text(
            '<!doctype html>\n<meta charset="utf-8">\n' + frame.to_html(),
            encoding='utf-8'
            )
        paths['html'] = path
        descriptor.update(html=str(path), mtime_ns=path.stat().st_mtime_ns)
    data[DF_MIME] = descriptor
    if paths:
        data['text/markdown'] = '**数据文件**\n\n' + '\n'.join(
            f'- [{fmt}: {path.name}](<{path.as_posix()}>)'
            for fmt, path in paths.items()
            )
    display(data, raw=True, **kwargs)
    return paths
