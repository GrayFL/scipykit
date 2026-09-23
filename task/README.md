# task：常用工具组合

这个包只聚合公共接口，具体功能实现在对应子包中。

| 入口 | 提供内容 |
| --- | --- |
| [task.cv](cv/README.md) | 全部绘图工具 + cv2 + 图像 IO，首次导入自动应用历史绘图 rcParams |
| [task.file](file/README.md) | 文件 IO、复制和目录树工具 |

```python
from scipykit.task.cv import *
fig = mfigure((6, 3), dpi=150, alpha=1)
disp(fig, "name")
savefig(fig, dpi=300, alpha=0)
plt.close(fig)
```

添加新组合只做导入和 `__all__` 声明。绘图工具见 [mtp_initializer](../mtp_initializer/README.md)，IO 工具见 [sci_initializer](../sci_initializer/README.md)，源码使用方式见 [仓库概览](../README.md)。
