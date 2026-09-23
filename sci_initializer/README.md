# sci_initializer：数据与文件 IO

提供 NumPy/pandas 快捷导入、Markdown 表格解析、JSON/pickle、目录操作，以及支持中文路径的 OpenCV 图像 IO。

入口依赖 NumPy 和 pandas；图像功能另需 OpenCV。Parquet、Excel 和内存统计的额外依赖按需安装，详见下文。

## 快速使用

```python
from scipykit.sci_initializer import *

root = Path("assets/data")
root.mkdir(parents=True, exist_ok=True)
config = {"实验": "exp01", "batch_size": 16}
json_save(config, root / "config.json", indent=2)
assert json_load(root / "config.json") == config

array = np.arange(6).reshape(2, 3)
pkl_save(array, root / "array.pkl")
restored = pkl_load(root / "array.pkl")
```

JSON/pickle 保存前需自行创建父目录；pickle 用于自己创建或可信来源的文件。旧聚合入口 `from scipykit.task.file import *` 继续可用。

## 数据与目录工具

| 工具 | 用途 |
| --- | --- |
| `np` / `pd` / `Path` | 常用交互别名 |
| `pd_read_markdown_table(text, missing_text="")` | 解析首尾带竖线的 Markdown 表格，保留空单元格，去除文本样式 |
| `json_save` / `json_load` | JSON 读写，保存参数透传 json.dump |
| `pkl_save` / `pkl_load` | pickle 读写 |
| `copy_to(src, dst)` | 将源文件或目录复制到目标目录内 |
| `generate_tree(path, ...)` | 生成目录树文本 |
| `sizeof(obj, backend="sys")` | 简单内存大小；默认 pympler 后端需相应依赖 |

## 图像 IO

```python
image = cv2_imread("图片.png", is_to_RGB=True)
cv2_imwrite("output/图片.png", image, is_from_RGB=True, comp_ratio=3)
```

`cv2_imread` 保留灰度及 alpha 通道；解码失败明确报错。`cv2_imwrite` 自动创建父目录，支持大小写扩展名；`comp_ratio` 为 0–9 的整数，PNG 越高压缩越大，JPEG 对应质量 100–10。

`cvt(img)` 默认交换红蓝通道，支持灰度和四通道；显式 `flag` 透传 OpenCV。画框、标注文字、画点及显示图像见 [mtp_initializer](../mtp_initializer/README.md)，需要一站式导入可用 [task.cv](../task/cv/README.md)。

## DataFrame 显示与 Parquet

```python
from scipykit.mtp_initializer import disp

frame = pd.DataFrame({"方法": ["基线", "改进"], "准确率": [0.82, 0.91]})
disp(frame)
paths = disp(frame, "metrics")
if "parquet" in paths:
    restored = pd.read_parquet(paths["parquet"])
else:
    restored = pd.read_csv(paths["csv"], index_col=0)
```

命名 DataFrame 默认保存数据和外置 HTML：安装 pyarrow 或 fastparquet 时选择 Parquet，没有引擎时自动回退到 CSV。`df_formats=("parquet", "csv", "xlsx")` 可同时保存多种格式；显式请求 Parquet 时需要对应引擎，XLSX 需要 pandas 支持的 Excel 写入引擎（例如 openpyxl）。表格显示实现位于绘图包 `display.py`；其私有 MIME 和返回路径均使用绝对路径。

实现按 `tables.py / files.py / images.py` 拆分。完整示例见 [数据与文件 IO](../examples/05_数据与文件IO.ipynb)，源码使用和示例依赖见 [仓库概览](../README.md)。
