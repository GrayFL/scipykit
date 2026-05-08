import numpy as np
import cv2
from pathlib import Path
from typing import Union, Literal

from scipykit.mtp_initializer import *

# ===== IMG PROCESS ===== #


def cv2_imread(file_path: Union[str, Path], is_to_RGB=False):
    """
    更安全的读取图片方式
    """
    cv_img = cv2.imdecode(
        np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
    if is_to_RGB:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img


def cv2_imwrite(
        file_path: Union[str, Path],
        img: np.ndarray,
        comp_ratio: int = None,
        is_from_RGB=False
    ):
    """
    更安全的保存图片方式

    Parameters
    ----------
    comp_ratio : int, optional
        压缩比，默认为None，即不压缩，若为整数，则为压缩比，范围为[0,9]，越大压缩率越高，文件越小  
        注：内部做了一次转换，对于 ``png`` 取值范围为[0,9]， 越大压缩率越高，对于 ``jpg`` 取值范围为[0,100]，越小压缩率越高，文件越小
    is_from_RGB : bool
        若原图像不是BGR空间，则需要设置为True
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ext = file_path.suffix[1:]
    if is_from_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if None is not comp_ratio:
        assert comp_ratio >= 0 and comp_ratio <= 9, \
            f'comp_ratio must be in [0,10], but got {comp_ratio}'
        if ext in ['jpg', 'jpeg']:
            ext = 'jpg'
            comp_ratio = int(100 - comp_ratio*10)
            compression_params = [cv2.IMWRITE_JPEG_QUALITY, comp_ratio]
        elif ext == 'png':
            ext = 'png'
            comp_ratio = int(comp_ratio)
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, comp_ratio]
        else:
            raise ValueError(f'Invalid extension: {ext}')
    else:
        compression_params = []
    ret, cv_img = cv2.imencode(f'.{ext}', img, compression_params)
    cv_img.tofile(file_path)
    return ret


def canvas_to_array(fig: plt.Figure, mode: Literal['RGB', 'RGBA'] = 'RGB'):
    img = np.array(fig.canvas.buffer_rgba())
    if mode == 'RGB':
        img = img[..., :3]
    elif mode == 'RGBA':
        img = img[..., :4]
    else:
        raise ValueError(f'Invalid mode: {mode}')
    return img


def cvt(img: np.ndarray, flag: int = cv2.COLOR_BGR2RGB) -> np.ndarray:
    """
    基于 opencv 做色彩空间转换
    """
    return cv2.cvtColor(img, flag)


# ===== Annotation ===== #


def draw_box(
        img: np.ndarray,
        outs: np.ndarray,
        color=(0, 255, 255),
        is_to_RGB=False
    ):
    """
    绘图不会改变原始图像
    outs:
        格式为[x1,x2,y1,y2]
    color:
        BGR
    """
    res = img.copy()
    for i, out in enumerate(outs):
        # color = (0,255,255)
        res = cv2.rectangle(
            res, out[0:2], out[2:4], color=color, thickness=2
            )
    if is_to_RGB:
        res = cvt(res)
    return res


def draw_text(
        img: np.ndarray,
        text: "list[str]",
        outs: np.ndarray,
        color=(0, 255, 255),
        fontScale=2,
        thickness=2,
        is_to_RGB=False,
        **kwds
    ):
    """
    绘图不会改变原始图像
    outs:
        格式为[x1,y1,x2,y2]
    color:
        BGR
    """
    res = img.copy()
    for i, out in enumerate(outs):
        # color = (0,255,255)
        res = cv2.putText(
            res,
            text[i], [int(out[0]), int(out[3])],
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            **kwds
            )
    if is_to_RGB:
        res = cvt(res)
    return res


def draw_pts(
        img: np.ndarray,
        pts: np.ndarray,
        color=(0, 255, 255),
        is_to_RGB=False
    ):
    '''
    pts: [[x,y], ...]
    '''
    res = img.copy()
    for i, pt in enumerate(pts):
        res = cv2.circle(res, pt, radius=20, color=color, thickness=-1)
    if is_to_RGB:
        res = cvt(res)
    return res


def mark_on_axes(
        ax: Axes,
        xyxys: np.ndarray,
        edgecolor: Union[str, tuple, list] = (1.0, 0.498, 0.055, 0.9),
        facecolor: Union[str, tuple, list] = (1.0, 0.733, 0.471, 0.4),
        # alpha: float = None,
        **kwds
    ):
    """
    在 Axes 上绘制标注框，注意并仅仅是绘制标注而不包含原图，因此需要在外部绘制原图

    多余的参数会原封不动传参给 plt.Rectangle()

    Parameters
    ----------
    ax : Axes
        绘图对象
    xyxys : np.ndarray
        格式为 [[x1,y1,x2,y2], ...]
    edgecolor : str | tuple
        边框颜色
    facecolor : str | tuple
        填充颜色，一般建议不使用边框颜色，只使用背景颜色。默认是橙色
    alpha : float
        透明度
    """
    for _i, bbox in enumerate(xyxys):
        ax.add_patch(
            plt.Rectangle(
                xy=(bbox[0], bbox[1]),
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
                edgecolor=edgecolor if isinstance(edgecolor,
                                                    (str, tuple)) or
                not hasattr(edgecolor, '__getitem__') else edgecolor[_i],
                facecolor=facecolor if isinstance(facecolor,
                                                    (str, tuple)) or
                not hasattr(facecolor, '__getitem__') else facecolor[_i],
                rotation_point='center',
                **kwds
                )
            )
        # ax.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]],
        #         [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]],
        #         color=edgecolor,
        #         **kwds)
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        head = ()


# ===== Color ===== #


def with_alpha(color: tuple, alpha: float = 1.0):
    """
    给颜色添加透明度，返回一个 RGBA 颜色
    """
    return (*color[:3], alpha)
