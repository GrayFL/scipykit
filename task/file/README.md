# task.file：文件快捷入口

```python
from scipykit.task.file import *

config = json_load("config.json")
print(generate_tree(".", max_depth=2))
```

保留 `Path`、`sizeof`、`pkl_save/pkl_load`、`json_save/json_load`、`copy_to`、`generate_tree` 的组合导入，函数实现统一位于 [sci_initializer.files](../../sci_initializer/README.md)。此入口不负责初始化绘图。

示例：[数据与文件 IO](../../examples/05_数据与文件IO.ipynb)。使用准备：[仓库概览](../../README.md)。
