# mtp_initializer：科研绘图

从 `scipykit.mtp_initializer` 或 `scipykit.task.cv` 首次导入，即自动应用历史 rcParams 默认配置。像以前一样直接开始画图即可。

绘图入口依赖 NumPy、Matplotlib 和 IPython；`configure_inline` 使用 matplotlib-inline。图像 IO 另需 OpenCV，表格导出依赖见 [sci_initializer](../sci_initializer/README.md)。

## 快速使用

```python
from scipykit.mtp_initializer import *
# CV 项目也可以继续使用：from scipykit.task.cv import *

# 导入已自动应用历史 rcParams 默认预设，无需额外启用样式
fig = mfigure((6, 3), dpi=150, alpha=0)
ax = fig.add_axes([0.13, 0.18, 0.82, 0.72])
ax.plot([0, 1, 2], [0, 1, 0], label="signal")
label_axes(ax, "x", "y", grid=True, legend=True)
disp(fig, "name") # 通常建议增加名字来落盘，少用 base64 直接编码进 notebook
savefig(fig, dpi=300, alpha=0)  # 沿用 name 的路径，透明导出
plt.close(fig)
```

`mfigure` 保留历史背景 Axes：`fig.axes[0]` 是背景，在其上添加数据 Axes。多子图可使用 `fig, axes = subplots(1, 2, layout="constrained")`，它返回标准 Matplotlib 对象。

| 工具 | 用途 |
| --- | --- |
| `plot_style("paper")` | 临时样式上下文；另有 notebook、presentation |
| `get_cmap(name, lut=None)`、`palette()`、`with_alpha()` | 色图、配色、透明度 |
| `plot_band(ax, x, y, std=...)` | 曲线与误差带，返回 line/band |
| `label_axes`、`panel_label`、`add_colorbar` | 轴标签、子图编号、色条 |
| `mark_on_axes(ax, boxes)` | xyxy 标注框，返回 Rectangle 列表 |
| `text_better` | 单个/批量文字自动避让，保留原有接口 |
| `canvas_to_array(fig, "RGBA")` | 自动绘制并复制为像素数组 |
| `show_image`、`draw_box`、`draw_text`、`draw_pts` | 图像显示、在副本上标注 |

`text_better(..., placement_mode="exact")` 适合任意字体；`mono_fast` 适合等宽字体组合。默认字体为 Inconsolata / Sarasa Mono SC，需在目标机器上安装，或通过 `set_plot_style(**{'font.family': ['已安装的字体名称']})` 替换；中文绘图需选择包含中文字形的字体。三维和极坐标不能使用二维单位换算函数；非线性直角轴使用 `at=(x, y)` 指定局部参考点。

## 显示与保存

- `disp(fig)`：直接显示。`disp(fig, "name")`：保存 `assets/<notebook>/name.png` 并显示外置资源；私有 notebook-assets MIME 中的 `image` 路径为绝对路径，同时提供标准 HTML 引用。
- 在不支持外置资源或需要分享 notebook 的前端，使用 `disp(fig, "name", embed=True)` 内嵌 PNG。只有内嵌图像能完全脱离 assets 目录分享。
- `savefig(fig)`：沿用最近一次 `disp(fig, key)` 的路径；没有命名时写 `assets/figure.png`。同一路径会覆盖，可显式传文件路径。`alpha=0` 默认使所有 Axes 背景透明，保存后原图不变。
- 环境变量 `NOTEBOOK_ASSETS_ROOT` 控制资源根目录；`NOTEBOOK_NAME` 手动指定子目录；也可给 `disp` 传 `assets_root`、`notebook_name`。未指定时尝试 VS Code notebook 名称，最后回退到 `notebook`。
- `disp(df)` 默认紧凑纯文本；`df_render_html=True` 启用未命名表格的内嵌 HTML。命名表格支持 `df_formats=("csv", "xlsx")`，完整 HTML 外置。私有 DataFrame MIME 的 `data` 各格式路径与 `html` 路径均为绝对路径，`disp` 返回的 Path 也保持绝对路径。
- `df_formats=None` 自动选择已安装的 Parquet 引擎，否则使用 CSV。Parquet 需要 pyarrow 或 fastparquet，XLSX 需要 pandas 支持的 Excel 写入引擎（例如 openpyxl）。也可显式指定 `df_formats=("parquet", "csv", "xlsx")`。

## 默认配置与可选样式

| rcParams | 默认值 |
| --- | --- |
| `font.family` | Inconsolata、Sarasa Mono SC |
| `font.size` / `mathtext.fontset` | 10.5 / stix |
| `figure.dpi` | 300 |
| `figure.subplot.left/right` | 0.11 / 0.89 |
| `figure.subplot.bottom/top` | 0.15 / 0.91 |
| `figure.facecolor` / `axes.facecolor` | `(1, 1, 1, 0)` |
| `axes.titlesize` / `legend.fontsize` | medium |
| `interactive` | False |

这些值集中定义在 `style.py` 的 `DEFAULT_RC_PARAMS` 中。默认预设名为 `default`；`paper` 额外隐藏上/右边框和图例边框，`notebook`、`presentation` 调整字号等。默认使用无需调用任何设置函数。

```python
# 需要定制时再设置；with 退出后恢复原样。
with plot_style("paper", **{"font.size": 9}):
    fig, ax = subplots()
    ax.plot([0, 1], [0, 1])
    disp(fig)
    plt.close(fig)

# 也可以持久调整会话配置：
set_plot_style("presentation")
# 重新应用历史默认值：set_plot_style()
```

Python 首次加载绘图包时执行初始化，直接 `import scipykit.mtp_initializer` 也会生效；重复 `import *` 遵循模块缓存，不覆盖之后的手动定制。包 reload 会重新应用默认预设。导入不创建或关闭 Figure，不更换 backend，也不覆盖资源目录环境变量。Notebook 输出格式可按需用 `configure_inline(("png",))` 配置。

## 模块与示例

实现拆为 `figure / display / style / colors / geometry / elements / images / annotator`，入口 `__all__` 提供便捷导出。图像 IO 使用 [sci_initializer](../sci_initializer/README.md)，组合入口见 [task.cv](../task/cv/README.md)。

- [绘图快速上手](../examples/01_绘图快速上手.ipynb)
- [图像与自动标注](../examples/02_图像与自动标注.ipynb)
- [仓库概览](../README.md)
