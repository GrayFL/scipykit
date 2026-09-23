"""图像展示与 OpenCV 标注；所有绘图实现集中于此包。"""
import numpy as np
import matplotlib.pyplot as plt
from .figure import mfigure, canvas_to_array
from .display import disp


def _rgb(img, enabled):
    if enabled:
        from scipykit.sci_initializer.images import cvt
        return cvt(img)
    return img


def show_image(img, scale=1.0, size_factor=120, is_output=False,
               is_to_RGB=False, key=None, **kwargs):
    """一次展示并关闭画布；is_output=True 返回独立 RGBA 数组。"""
    if scale <= 0 or size_factor <= 0:
        raise ValueError('scale 和 size_factor 必须大于零')
    img = np.asarray(img)
    h, w = img.shape[:2]
    fig = mfigure((w / size_factor, h / size_factor), dpi=size_factor * scale, frame=False)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(_rgb(img, is_to_RGB), **kwargs)
        disp(fig, key=key)
        if is_output:
            return canvas_to_array(fig, 'RGBA')
    finally:
        plt.close(fig)


def draw_box(img, outs, color=(0, 255, 255), is_to_RGB=False, *, thickness=2):
    """在图像副本上画框；outs 为 [x1,y1,x2,y2]。"""
    import cv2
    result = img.copy()
    for x1, y1, x2, y2 in np.asarray(outs).reshape(-1, 4):
        cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return _rgb(result, is_to_RGB)


def draw_text(img, text, outs, color=(0, 255, 255), fontScale=2,
              thickness=2, is_to_RGB=False, **kwargs):
    """使用 OpenCV 内置字体，在 xyxy 框左下方标注；中文请用 text_better。"""
    import cv2
    boxes = np.asarray(outs).reshape(-1, 4)
    labels = [text] * len(boxes) if isinstance(text, str) else list(text)
    if len(labels) != len(boxes):
        raise ValueError('text 与 outs 数量必须相同')
    result = img.copy()
    font = kwargs.pop('fontFace', cv2.FONT_HERSHEY_PLAIN)
    for label, (x1, y1, x2, y2) in zip(labels, boxes):
        cv2.putText(result, str(label), (int(x1), int(y2)), font,
                    fontScale, color, thickness, **kwargs)
    return _rgb(result, is_to_RGB)


def draw_pts(img, pts, color=(0, 255, 255), is_to_RGB=False, *, radius=20, thickness=-1):
    """在图像副本上画点。"""
    import cv2
    result = img.copy()
    for x, y in np.asarray(pts).reshape(-1, 2):
        cv2.circle(result, (int(x), int(y)), radius, color, thickness)
    return _rgb(result, is_to_RGB)
