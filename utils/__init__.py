import numpy as np
import cv2
import json
import pickle
from typing import Union, Literal

# ===== IMG PROCESS ===== #


def cv2_imread(file_path, is_RGB=False):
    """
    更安全的读取图片方式
    """
    cv_img = cv2.imdecode(
        np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    if is_RGB:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img


def cv2_imwrite(file_path: str, img: np.ndarray, is_RGB=False):
    """
    更安全的保存图片方式

    ---

    is_RGB: 是否需要转换为RGB格式保存
    """
    ext = file_path.split('.')[-1]
    ret, cv_img = cv2.imencode(f'.{ext}', img)
    cv_img.tofile(file_path)
    # if is_RGB:
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return ret


def cvt(img: np.ndarray) -> np.ndarray:
    """
    将opencv读入的BGR转换为RGB空间
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


# ===== FILE ===== #


def sizeof(var: object, backend: Literal['pympler', 'sys'] = 'pympler'):
    if 'pympler' == backend:
        from pympler.asizeof import asizeof
        size = asizeof(var)
    elif 'sys' == backend:
        from sys import getsizeof
        size = getsizeof(var)
    if (size // 1024) == 0:
        return f'{size} B'
    elif (size // 1024**2) == 0:
        return f'{size/1024:.2f} KB'
    elif (size // 1024**3) == 0:
        return f'{size/1024**2:.2f} MB'


def pkl_save(obj, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(file_path: str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def json_save(obj, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(obj, f)


def json_load(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)
