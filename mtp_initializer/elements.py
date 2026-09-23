"""常见绘图元素；返回原生 Artist，方便继续定制。"""
import numpy as np
from matplotlib.colors import is_color_like
from matplotlib.patches import Rectangle


def mark_on_axes(
        ax,
        xyxys,
        edgecolor=(1, .498, .055, .9),
        facecolor=(1, .733, .471, .4),
        **kwargs
    ):
    """绘制 xyxy 标注框；颜色支持单色或每框一色，返回 Rectangle 列表。"""
    boxes = np.asarray(xyxys).reshape(-1, 4)
    artists = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        patch = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            edgecolor=edgecolor
            if is_color_like(edgecolor) else edgecolor[i],
            facecolor=facecolor
            if is_color_like(facecolor) else facecolor[i],
            **kwargs,
            )
        ax.add_patch(patch)
        artists.append(patch)
    return artists


def label_axes(
        ax,
        xlabel=None,
        ylabel=None,
        title=None,
        *,
        grid=False,
        legend=False
    ):
    """设置常用轴标签和网格，返回 Axes。"""
    for name, value in [('xlabel', xlabel), ('ylabel', ylabel), ('title', title)]:
        if value is not None:
            getattr(ax, 'set_' + name)(value)
    ax.grid(grid, **({'alpha': .2} if grid else {}))
    if legend:
        ax.legend()
    return ax


def panel_label(ax, label, xy=(-.12, 1.04), **kwargs):
    """添加 (a)/(b) 等子图编号。"""
    defaults = dict(transform=ax.transAxes, weight='bold', va='bottom')
    return ax.text(*xy, label, **{**defaults, **kwargs})


def plot_band(
        ax,
        x,
        y,
        lower=None,
        upper=None,
        *,
        std=None,
        color=None,
        label=None,
        band_alpha=.2,
        **kwargs
    ):
    """曲线及误差带；显式上下界或 y±std，返回 (Line2D, PolyCollection)。"""
    y = np.asarray(y)
    if std is not None:
        if lower is not None or upper is not None:
            raise ValueError('std 与 lower/upper 不能同时传入')
        lower, upper = y - np.asarray(std), y + np.asarray(std)
    if lower is None or upper is None:
        raise ValueError('请提供 std 或 lower/upper')
    line, = ax.plot(x, y, color=color, label=label, **kwargs)
    band = ax.fill_between(
        x,
        lower,
        upper,
        color=line.get_color(),
        alpha=band_alpha,
        linewidth=0
        )
    return line, band


def add_colorbar(mappable, ax=None, *, label=None, **kwargs):
    """为图像/散点图添加色条，返回 Colorbar。"""
    ax = ax if ax is not None else mappable.axes
    bar = ax.figure.colorbar(mappable, ax=ax, **kwargs)
    if label is not None:
        bar.set_label(label)
    return bar
