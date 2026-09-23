# scipykit

通用 Python 科研工具库，打包科研绘图、图像处理、基础 IO 和随手日志。

当前按源码目录使用：将 `scipykit/` 的父目录加入所选 Python 环境的 `PYTHONPATH` 或 `.pth` 文件，并按各子包文档准备依赖。路径和环境由使用者配置。

## 子包文档

| 子包 | 用途 |
| --- | --- |
| [mtp_initializer](mtp_initializer/README.md) | 科研绘图、自动标注、图像显示与外置资源；导入即应用历史 rcParams |
| [sci_initializer](sci_initializer/README.md) | 数据、JSON/pickle、目录工具、OpenCV 图像 IO，可选 Parquet 支持 |
| [log](log/README.md) | 分级日志、主题、来源定位、按需落盘 |
| [task](task/README.md) | 常用工具组合：[cv](task/cv/README.md)、[file](task/file/README.md) |

```python
from scipykit.mtp_initializer import *  # 也可 from scipykit.task.cv import *
fig = mfigure((6, 3), dpi=150, alpha=0)
disp(fig, "name")
savefig(fig, dpi=300, alpha=0)
```

## 示例与维护

中文 notebook：[绘图](examples/01_绘图快速上手.ipynb)、[图像与标注](examples/02_图像与自动标注.ipynb)、[日志入门](examples/03_日志快速上手.ipynb)、[日志定制](examples/04_日志主题与格式.ipynb)、[数据 IO](examples/05_数据与文件IO.ipynb)。选择已安装所需依赖的 Python 内核，从上到下运行；开头统一设置 autoreload 和项目路径。

运行完整示例需准备各维护子包的依赖，以及 Jupyter/IPython、pyarrow、openpyxl；自动执行另需 nbformat、nbclient、jupyter_client 和 ipykernel。测试使用 pytest。在仓库根目录、所选环境中执行：

```bash
python -m pytest -q
python tests/run_notebooks.py
```

## 历史包

仍保留源码的历史目录：[evaluation](evaluation/README.md)、[visualization](visualization/README.md)。它们未纳入维护模块的兼容性测试；新功能优先使用上面的维护子包。

本机调试环境和 coding agent 开发上下文集中记录于根目录的 `PROJECT.md`。该文件为本地开发文档，不随包分发。
