import numpy as np
import pytest
from scipykit.task.cv import cv2_imread, cv2_imwrite, draw_box, draw_text, draw_pts, cvt


@pytest.mark.parametrize('channels', [None, 3, 4])
def test_chinese_path_and_color_roundtrip(tmp_path, channels):
    shape = (16, 20) if channels is None else (16, 20, channels)
    img = np.random.default_rng(3).integers(0, 256, shape, dtype=np.uint8)
    path = tmp_path / '中文目录' / '图.PNG'
    assert cv2_imwrite(path, img, comp_ratio=3, is_from_RGB=True)
    np.testing.assert_array_equal(cv2_imread(path, is_to_RGB=True), img)
    np.testing.assert_array_equal(cvt(cvt(img)), img)


def test_bad_images_and_compression_fail_explicitly(tmp_path):
    path = tmp_path / 'broken.png'
    path.write_text('invalid')
    with pytest.raises(ValueError):
        cv2_imread(path)
    with pytest.raises(FileNotFoundError):
        cv2_imread(tmp_path / 'missing.png')
    with pytest.raises(ValueError):
        cv2_imwrite(path, np.zeros((5, 5, 3), np.uint8), comp_ratio=10)


def test_drawing_copies_and_accepts_float_coordinates():
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    boxes = [[2.5, 3.1, 20.8, 21.0]]
    assert draw_box(img, boxes).sum() > 0
    assert draw_text(img, ['A'], boxes, fontScale=1).sum() > 0
    assert draw_pts(img, [[10.2, 12.5]], radius=3).sum() > 0
    assert img.sum() == 0
