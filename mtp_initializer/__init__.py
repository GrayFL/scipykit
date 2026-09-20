"""
关于字号和磅的对应：

| 中文字号 | 英文字号（磅）/pt |
|----------|-------------------|
| 初号     | 42                |
| 小初     | 36                |
| 一号     | 26                |
| 小一     | 24                |
| 二号     | 22                |
| 小二     | 18                |
| 三号     | 16                |
| 小三     | 15                |
| 四号     | 14                |
| 小四     | 12                |
| 五号     | 10.5              |
| 小五     | 9                 |
| 六号     | 7.5               |
| 小六     | 6.5               |
| 七号     | 5.5               |
| 八号     | 5                 |
"""

import os
from matplotlib.cm import get_cmap as mtp_get_cmap
import matplotlib as mtp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.gridspec import GridSpec
from IPython.display import display, display_markdown, clear_output as clr, Image as IPyImage
from IPython.core.formatters import format_display_data
# plt.rcParams['font.family'] = 'Times New Roman,Simsun'
plt.rcParams['font.family'] = 'Inconsolata, Sarasa Mono SC'
plt.rcParams['font.size'] = 10.5  # 10.5pt 五号字；9pt 小五号字
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.subplot.left'] = 0.11  #0.05
plt.rcParams['figure.subplot.right'] = 0.89  #0.98
plt.rcParams['figure.subplot.bottom'] = 0.15  #0.1
plt.rcParams['figure.subplot.top'] = 0.91  #0.93
plt.rcParams['figure.facecolor'] = (1, 1, 1, 0)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = (1, 1, 1, 0)
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['interactive'] = 'False'
# plt.rcParams['color_cycle'] = ['b','g','r','c','m','y','k']
import numpy as np
from pathlib import Path

from .anotator import (
    text_better,
    find_annotate_position,
    find_annotate_positions,
    select_artists,
    clear_mono_metric_cache,
    )

# import matplotlib.backends.backend_agg
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats(
    *['png', 'jpeg'],  # *['png', 'jpeg', 'svg']
    # *args 表示激活哪些格式的输出（也就是vscode里的更改演示文稿，默认只打开png），'jpg'、'jpeg'等效
    bbox_inches=None,  # 默认是jupyter会设置为tight，会在mfigure外产生空白，所以要设置为 None
    format='png',  # 默认的display格式
    pil_kwargs={
        'quality': 80,  # 图片质量，0-100，越大质量越好。只能对JPEG格式生效
        'optimize': True,  # 对GIF的优化
        'compress_level': 9,  # PNG的压缩级别，0-9，越大压缩越好。但似乎不起作用
        },
    )

plt.plot(0, 0)
plt.ioff()
plt.close()

from typing import Literal

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['NOTEBOOK_ASSETS_ROOT'] = 'assets'


def get_cmap(name: Literal['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cividis', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'inferno', 'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'tab10', 'tab20', 'tab20b', 'tab20c', 'terrain', 'twilight', 'twilight_shifted', 'viridis', 'winter'], **kwds): #yapf: disable
    """
    Parameters
    ---
    name:
        - 预设热力图 `'viridis', 'plasma', 'inferno', 'magma', 'cividis'`
        - 同色系亮度递减 `'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'`
        - 亮度变化更剧烈 `'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'`
        - 中间为白色 `'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'`
        - 环形色相 `'twilight', 'twilight_shifted', 'hsv'`
        - 量化色盘 `'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'`
        - 其他 `'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar'`

    Others
    ---
    'Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cividis', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'inferno', 'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'tab10', 'tab20', 'tab20b', 'tab20c', 'terrain', 'twilight', 'twilight_shifted', 'viridis', 'winter'
    """
    return mtp_get_cmap(name, **kwds)


def mfigure(
        figsize=(6.66, 3),
        dpi=300,
        frame=True,
        face_color=(1, 1, 1, 0),
        alpha=0,
        **kdic
    ):
    fig = plt.figure(figsize=figsize, dpi=dpi, **kdic)
    fig.set_facecolor((1, 1, 1, alpha))
    bed: Axes = fig.add_axes([0, 0, 1, 1])
    bed.set_frame_on(frame)
    bed.set_xticks([])
    bed.set_yticks([])
    bed.set_facecolor(face_color)
    return fig


def disp(
        obj,
        key=None,
        df_formats=('parquet', ),
        df_render_html=False,
        df_link_html=True,
        **kwds
    ):
    if isinstance(obj, plt.Figure):
        if key is not None:
            root = Path(os.environ["NOTEBOOK_ASSETS_ROOT"])
            p = root / inspect_notebook_name() / f"{key}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            obj.savefig(p)
            v = p.stat().st_mtime_ns
            IMAGE_MIME = "application/vnd.notebook-assets.image+json"
            # display(IPyImage(url=f"{p}?mtime={v}"), **kwds)
            display({
                IMAGE_MIME: {
                    "kind": "image", "image": str(p)
                    },
                "text/plain": f"{p}"
                },
                    raw=True)
            # print(p)
            return
        display(obj, **kwds)
        return

    import pandas as pd
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        frame = (obj.to_frame() if isinstance(obj, pd.Series) else obj)

        # 给 notebook / coding agent 使用的紧凑文本表示。
        with pd.option_context(
                "display.expand_frame_repr",
                False,
            ):
            plain_text = repr(frame)

        output_data = {
            "text/plain": plain_text,
            }
        # ------------------------------------------------------------
        # 无 key：
        #   默认完全不产生 HTML，只保留 text/plain。
        #
        #   df_render_html=True 时才恢复传统 notebook HTML rendering，
        #   此时 HTML 本身会进入 ipynb。
        # ------------------------------------------------------------
        if key is None:
            if df_render_html:
                data, _ = format_display_data(
                    obj,
                    include=["text/html"],
                )
                html = data.get("text/html")

                if html is None:
                    html_obj = (
                        obj.to_frame()
                        if isinstance(obj, pd.Series) else obj
                        )
                    html = html_obj.to_html(
                        max_rows=pd.get_option("display.max_rows"),
                        max_cols=1e2,  # pd.get_option("display.max_columns")
                        show_dimensions=True,
                        )

                output_data["text/html"] = html

            display(
                output_data,
                raw=True,
                **kwds,
                )
            return
        # ------------------------------------------------------------
        # 有 key：本地化 dataframe。
        # ------------------------------------------------------------
        root = Path(os.environ["NOTEBOOK_ASSETS_ROOT"])
        asset_dir = root / inspect_notebook_name()
        asset_dir.mkdir(parents=True, exist_ok=True)
        data_paths = {}
        for fmt in df_formats:
            p = asset_dir / f"{key}.{fmt}"
            if fmt == "parquet":
                frame.to_parquet(p)
            elif fmt == "csv":
                frame.to_csv(
                    p,
                    header=True,
                    )
            elif fmt == "xlsx":
                frame.to_excel(
                    p,
                    header=True,
                    index=False,
                    )
            else:
                raise ValueError(f"Unsupported dataframe format: {fmt!r}")
            data_paths[fmt] = p
        DF_MIME = ("application/vnd.notebook-assets.dataframe+json")
        descriptor = {
            "kind":
                ("series" if isinstance(obj, pd.Series) else "dataframe"),
            "shape": str(' x '.join(map(str, frame.shape))),
            # "data": {
            #     fmt: str(p)
            #     for fmt, p in data_paths.items()
            #     },
            }
        # ------------------------------------------------------------
        # 外部 HTML：
        #
        # df_link_html=True:
        #   保存完整 HTML 到 assets。
        #
        # notebook 中的 text/html 只保存一个非常小的 link shell，
        # 不保存 dataframe 的 HTML 内容。
        #
        # 注意：
        #   key 存在时，df_render_html 不会把 dataframe HTML 注入
        #   notebook；外部 HTML 完全由 df_link_html 控制。
        # ------------------------------------------------------------
        if df_link_html:
            html_path = asset_dir / f"{key}.html"
            data, _ = format_display_data(
                obj,
                include=["text/html"],
            )
            html = data.get("text/html")
            if html is None:
                html_obj = (
                    obj.to_frame() if isinstance(obj, pd.Series) else obj
                    )
                html = html_obj.to_html(
                    max_rows=pd.get_option("display.max_rows"),
                    max_cols=pd.get_option("display.max_columns"),
                    show_dimensions=True,
                    )
            html = ("<!doctype html>\n"
                    '<meta charset="utf-8">\n' + html)
            html_path.write_text(
                html,
                encoding="utf-8",
                )
            descriptor["html"] = str(html_path)
            descriptor["mtime_ns"] = (html_path.stat().st_mtime_ns)
            # 这里注入 notebook 的不是 dataframe HTML，
            # 只是一小段链接 UI。
            import html as _html
            html_href = _html.escape(
                html_path.as_posix(),
                quote=True,
                )
            output_data[DF_MIME] = (descriptor)
        # ------------------------------------------------------------
        # 外部数据文件 presentation。
        #
        # VS Code 没有原生 Parquet notebook renderer，
        # 所以这里用 text/markdown 做一个可切换的文件 presentation。
        # 文件本身仍然完全在 ipynb 外部。
        # ------------------------------------------------------------
        if data_paths:
            markdown = [
                "**Localized dataframe files**",
                "",
                ]
            for fmt, p in data_paths.items():
                path = p.as_posix()
                markdown.append(f"- **{fmt}**: "
                                f"[`{p.name}`]({path})")
            output_data["text/markdown"] = "\n".join(markdown)
        # metadata = {
        #     "notebook-assets": descriptor,
        #     }
        display(
            output_data,
            raw=True,
            # metadata=metadata,
            **kwds,
            )
        return
    display(obj, **kwds)


def show_image(
        img,
        scale=1.0,
        size_factor=120,
        is_output=False,
        is_to_RGB=False,
        key=None,
        **kwds
    ):
    '''
    简单地，一次性地，在jupyter中绘制一张图片

    Parameters
    ----------
    scale : folat
        默认情况下采用dpi=120=size_factor，即1920x1080 -> 16x9; 通过scale可以等比例控制画布大小
    '''
    h, w, *_ = img.shape
    if is_to_RGB:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dpi = size_factor * scale
    fig = mfigure(
        figsize=(w / size_factor, h / size_factor), dpi=dpi, frame=False
        )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, **kwds)
    disp(fig, key=key)
    if is_output:
        fig.canvas.draw()
        frame = fig.canvas.buffer_rgba()
        plt.close(fig)
        return frame
    plt.close(fig)


def savefig(
        fig: plt.Figure,
        file_path: str,
        is_frame_on=False,
        alpha=1.0,
        **kwds
    ):
    """
    注意 该函数仅仅适用于使用mfigure创建的fig
    """
    fig.get_axes()[0].set_frame_on(is_frame_on)
    fig.set_facecolor((1, 1, 1, alpha))
    fig.savefig(file_path, **kwds)


def clipfig(fig, ext='jpg', **kwds):
    savefig(fig, f'./__temp__.{ext}', **kwds)
    import subprocess
    args = ['powershell', f'Get-Item ./__temp__.{ext} | Set-Clipboard']
    subprocess.Popen(args=args)


def xywh2xyxy(xywh: list):
    return [xywh[0], xywh[1], xywh[2] + xywh[0], xywh[3] + xywh[1]]


def xyxy2xywh(xywh: list):
    return [xywh[0], xywh[1], xywh[2] - xywh[0], xywh[3] - xywh[1]]


def get_pixels_per_data_unit(ax: Axes, invert=False):
    """Calculate pixels per data unit in x and y axes."""
    # Transform (0,0) and (1,0)/(0,1) to pixel coordinates
    input_dims = ax.transData.input_dims
    tmp = input_dims - 1
    tar = ([0] * tmp + [1]) * 2
    if not invert:
        func_transtorm = ax.transData.transform
    else:
        func_transtorm = ax.transData.inverted().transform
    ori_0 = func_transtorm([0] * input_dims)
    tar_axis = []
    for i in range(input_dims):
        tar_axis.append(func_transtorm(tar[i:i + input_dims])[i])

    d_axis = (np.array(tar_axis)[::-1] - ori_0)[::-1]
    return d_axis


def scale_data2pt(ax: Axes, data: float):
    """
    计算数据坐标系中的长度在pt中的长度
    """
    d_axis = get_pixels_per_data_unit(ax)
    dpi = ax.figure.dpi
    size_in_pt_xyz = data * d_axis * (72/dpi)
    return size_in_pt_xyz


def scale_pt2data(ax: Axes, pt: float):
    """
    计算pt在数据坐标系中的长度
    """
    d_axis = get_pixels_per_data_unit(ax, invert=True)
    dpi = ax.figure.dpi
    size_in_data_xyz = pt * (dpi/72) * d_axis
    return size_in_data_xyz


def inspect_notebook_name():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ns = ip.user_ns
            if "__vsc_ipynb_file__" in ns:
                return Path(ns["__vsc_ipynb_file__"]
                           ).expanduser().resolve().name
    except Exception:
        pass

    if "__vsc_ipynb_file__" in globals():
        return Path(globals()["__vsc_ipynb_file__"]
                   ).expanduser().resolve().name

    if '__notebook_filename__' in os.environ:
        return os.environ['__notebook_filename__']

    print('[Scipykit] Not run in vscode notebook')
    return 'notebook'
