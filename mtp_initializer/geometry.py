"""坐标与单位转换。非线性坐标可通过 at 指定局部参考点。"""
import numpy as np


def xywh2xyxy(xywh):
    return [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]


def xyxy2xywh(xyxy):
    return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]


def get_pixels_per_data_unit(ax, invert=False, *, at=None):
    """返回 x/y 方向的局部像素比例，反向轴保留符号。

    线性直角坐标精确；非线性轴采用局部数值微分；不支持三维/极坐标。
    invert=True 返回其倒数（数据单位/像素）。
    """
    if ax.name != 'rectilinear':
        raise ValueError('单位转换仅支持二维直角坐标 Axes')
    ax.figure.canvas.draw()
    point = np.asarray(at if at is not None else
                       ax.transData.inverted().transform(ax.bbox.get_points().mean(axis=0)), dtype=float)
    step = np.maximum(np.abs(point), 1) * 1e-6
    base = ax.transData.transform(point)
    scale = np.array([(ax.transData.transform(point + np.eye(2)[i] * step[i])[i] - base[i]) / step[i]
                      for i in range(2)])
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError('无法在此参考点计算单位比例')
    return 1 / scale if invert else scale


def scale_data2pt(ax, data, *, at=None):
    return np.asarray(data) * get_pixels_per_data_unit(ax, at=at) * 72 / ax.figure.dpi


def scale_pt2data(ax, pt, *, at=None):
    return np.asarray(pt) * get_pixels_per_data_unit(ax, invert=True, at=at) * ax.figure.dpi / 72
