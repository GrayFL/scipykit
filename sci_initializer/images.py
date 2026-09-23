"""支持中文路径的 OpenCV IO；OpenCV 在实际使用时导入。"""
from pathlib import Path
import numpy as np


def _swap_rb(img):
    import cv2
    if img.ndim == 2:
        return img
    channels = img.shape[-1]
    if channels not in (3, 4):
        raise ValueError('颜色转换要求灰度、三通道或四通道图像')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB if channels == 3 else cv2.COLOR_BGRA2RGBA)


def cv2_imread(file_path, is_to_RGB=False):
    """保留灰度/透明通道读取；无法解码时明确报错。"""
    import cv2
    data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        raise ValueError(f'无法解码图像: {file_path}')
    return _swap_rb(img) if is_to_RGB else img


def cv2_imwrite(file_path, img, comp_ratio=None, is_from_RGB=False):
    """保存图像并创建父目录；PNG 压缩 0–9，JPEG 映射为质量 100–10。"""
    import cv2
    path = Path(file_path)
    ext = path.suffix.lower()
    params = []
    if comp_ratio is not None:
        if not isinstance(comp_ratio, (int, np.integer)) or not 0 <= comp_ratio <= 9:
            raise ValueError('comp_ratio 必须是 0–9 的整数')
        if ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(comp_ratio)]
        elif ext in ('.jpg', '.jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, 100 - int(comp_ratio) * 10]
        else:
            raise ValueError(f'不支持此格式的 comp_ratio: {ext}')
    if is_from_RGB:
        img = _swap_rb(img)
    ok, encoded = cv2.imencode(ext, img, params)
    if not ok:
        raise OSError(f'图像编码失败: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
    return True


def cvt(img, flag=None):
    """显式 flag 透传 OpenCV；省略时交换 RGB/BGR（支持灰度与 alpha）。"""
    import cv2
    return _swap_rb(img) if flag is None else cv2.cvtColor(img, flag)
