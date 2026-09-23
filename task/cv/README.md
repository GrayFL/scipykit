# task.cv：计算机视觉快捷入口

```python
from scipykit.task.cv import *

image = cv2_imread("图片.png", is_to_RGB=True)
show_image(image)
```

导出 [mtp_initializer](../../mtp_initializer/README.md) 的全部公共绘图接口，以及 `cv2`、`cv2_imread`、`cv2_imwrite`、`cvt`。首次导入自动应用历史 rcParams，无需调用样式初始化函数。图像 IO 的实现在 [sci_initializer](../../sci_initializer/README.md)。

旧 CameraTools3D/Frustum 实现已移除，此入口提供绘图与图像 IO 工具；相机几何功能请使用独立的 graphmap 库。

示例：[图像与自动标注](../../examples/02_图像与自动标注.ipynb)。使用准备：[仓库概览](../../README.md)。
