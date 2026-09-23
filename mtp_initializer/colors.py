"""色图及可复用配色。"""
import matplotlib as mpl
from matplotlib.colors import to_rgba
import numpy as np

PALETTES = {
    'okabe_ito': ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000'),
    'muted': ('#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4', '#8C613C'),
    'bright': ('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377'),
}


def get_cmap(name=None, lut=None):
    """适配 Matplotlib 新色图库 API，保留旧 lut 参数。"""
    cmap = mpl.colormaps.get_cmap(name)
    return cmap if lut is None else cmap.resampled(lut)


mtp_get_cmap = get_cmap


def with_alpha(color, alpha=1.0):
    """颜色名称、十六进制或 RGB(A) -> RGBA。"""
    return to_rgba(color, alpha)


def palette(name='okabe_ito', n=None):
    """取预设颜色或从任意 Matplotlib 色图等距采样。"""
    if n is not None and (not isinstance(n, int) or n < 1):
        raise ValueError('n 必须是正整数')
    if name in PALETTES:
        colors = PALETTES[name]
        return list(colors) if n is None else [colors[i % len(colors)] for i in range(n)]
    return list(get_cmap(name)(np.linspace(0, 1, n or 8)))
