"""画布创建、像素导出和无副作用保存。"""
from pathlib import Path
import os
import sys
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def mfigure(
        figsize=(6.66, 3),
        dpi=150,
        frame=True,
        face_color=(1, 1, 1, 0),
        alpha=0,
        **kwargs
    ):
    """兼容旧背景 Axes；数据图仍可用 fig.add_axes/add_subplot 添加。"""
    with plt.ioff():
        fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
    fig.set_facecolor((1, 1, 1, alpha))
    bed = fig.add_axes([0, 0, 1, 1], label='_scipykit_background')
    bed.set_frame_on(frame)
    bed.set(xticks=[], yticks=[], facecolor=face_color)
    bed.set_in_layout(False)
    bed.set_zorder(-100)
    return fig


def subplots(
        nrows=1,
        ncols=1,
        *,
        figsize=(6.66, 3),
        dpi=150,
        alpha=1,
        **kwargs
    ):
    """标准 Matplotlib 布局，不添加背景 Axes，返回 (fig, axes)。"""
    with plt.ioff():
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **kwargs)
    fig.patch.set_alpha(alpha)
    return fig, axes


def savefig(fig, file_path=None, is_frame_on=False, alpha=0.0, **kwargs):
    """路径省略时沿用 disp(fig, key) 的路径，否则保存 assets/figure.png。

    alpha=0 默认透明导出所有 Axes 背景；不改变原图属性。返回 Path，
    文件对象返回自身。显式 transparent 参数优先。
    """
    target = file_path
    if target is None:
        target = getattr(fig, '_scipykit_path', None)
        if target is None:
            target = Path(
                os.environ.get('NOTEBOOK_ASSETS_ROOT', 'assets')
                ) / 'figure.png'
    if isinstance(target, (str, os.PathLike)):
        target = Path(target).expanduser()
        if not target.suffix:
            target = target.with_suffix('.' + kwargs.get('format', 'png'))
        target.parent.mkdir(parents=True, exist_ok=True)
    beds = [
        ax for ax in fig.axes if ax.get_label() == '_scipykit_background'
        ]
    frames = [ax.get_frame_on() for ax in beds]
    try:
        for ax in beds:
            ax.set_frame_on(is_frame_on)
        kwargs.setdefault('transparent', alpha == 0)
        kwargs.setdefault(
            'facecolor',
            'none' if kwargs['transparent'] else
            to_rgba(fig.get_facecolor(), alpha)
            )
        fig.savefig(target, **kwargs)
    finally:
        for ax, state in zip(beds, frames):
            ax.set_frame_on(state)
    return target


def canvas_to_array(fig, mode='RGB'):
    """先绘制再返回独立 uint8 数组；适用于尚未显示的画布。"""
    if mode not in ('RGB', 'RGBA'):
        raise ValueError(f'无效模式: {mode}')
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    original = fig.canvas
    try:
        canvas = original if hasattr(original, 'buffer_rgba'
                                    ) else FigureCanvasAgg(fig)
        canvas.draw()
        return np.array(canvas.buffer_rgba()
                       )[..., :3 if mode == 'RGB' else 4].copy()
    finally:
        fig.set_canvas(original)


def clipfig(fig, ext='jpg', **kwargs):
    """保留 Windows 文件剪贴板接口；其他系统请使用 savefig。"""
    if sys.platform != 'win32':
        raise NotImplementedError('clipfig 仅支持 Windows；请使用 savefig')
    with tempfile.NamedTemporaryFile(suffix='.' + ext,
                                        delete=False) as file:
        path = Path(file.name)
    savefig(fig, path, **kwargs)
    subprocess.run([
        'powershell',
        '-NoProfile',
        '-Command',
        "Get-Item -LiteralPath '" + str(path).replace("'", "''")
        + "' | Set-Clipboard"
        ],
                    check=True)
    return path
