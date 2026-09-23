# log：随手日志

基于 Loguru 和 Rich，默认只输出终端，支持级别、配色、业务上下文、真实调用来源和按需落盘。

## 快速使用


```python
from scipykit.log import *

log("临时查看", {"loss": 0.12})
log.warning("样本偏少", count=12)
log.success("训练完成", file="logs/result.log")  # 仅本句也落盘

with record_to("logs/stage.log"):
    log("记录这个阶段")
    log("本句只输出终端", file=False)

log_to("logs/run.log", rotation="10 MB", retention="7 days")
log("开始持续落盘")
stop_log()  # 停止持续记录，保留终端输出
```

`log` 和 `lprint` 等价。支持 TRACE、DEBUG、INFO、SUCCESS、WARNING、ERROR、CRITICAL；默认终端级别为 INFO，文件为 TRACE。多个位置参数按 `sep` 拼接，**不做 Loguru 的隐式花括号插值**，推荐使用 f-string。`pretty=False` 使用普通字符串表示；`log.exception(...)` 在 except 中记录完整 traceback。

```python
configure_log(preset="source", theme="pastel", level="DEBUG", colors=True)
log("[green]完成[/green]，继续下一步", markup=True)
log("单句强调", style="bold white on blue")

worker = get_logger("dataset", run="exp01")
with log_context(stage="validation"):
    worker.info("读取批次", batch=3)

with record_to("logs/metrics.jsonl", serialize=True):
    worker.success("评估完成", accuracy=0.93)
```

默认记录真实的模块、文件、函数和行号，`component/run/stage` 等手动字段标记业务职责；`depth=1` 可让自己封装的日志函数向外多追溯一层。上下文优先级为本句字段 > `bind/get_logger` > `log_context`。JSONL 包含 `time/level/message/module/file/function/line/process/thread/extra/exception`。

| 配置 | 可选值 |
| --- | --- |
| `preset` | minimal、compact、source、detailed |
| `theme` | default、pastel、light、mono，或 `{级别: Rich 样式}` 字典 |
| `format` | 自定义 record 格式，如 `{time:HH:mm:ss} {level.name} {extra[scope]}{message}` |
| `colors` | None 自动检测；True 强制颜色；False 关闭 |
| `stream`、`width`、`terminal` | 输出流、复杂对象排版宽度、是否启用终端 |

可直接传 Rich Table/Panel（保留文字布局，统一日志配色）和 Rich Text（保留细粒度样式）。标记默认按字面输出，需显式 `markup=True` 才解析。文本/JSON 文件统一 UTF-8，并清理正文、字段及异常文本中的 ANSI/OSC 控制码。

`configure_log` 可重复调用，不重复注册终端，也不干扰宿主项目的 `loguru.logger`。`record_to` 支持嵌套、线程和 asyncio 上下文隔离；创建的异步子任务需在退出上下文前等待完成。`log_to` 对同一路径幂等。文件 sink 接受 Loguru 的 `rotation/retention/compression/enqueue/mode` 等参数；修改已开启文件的参数前应先停止记录。多进程集中收集需在应用层配置，独立进程请使用各自的日志文件。

## 示例与实现位置

- [日志快速上手](../examples/03_日志快速上手.ipynb)
- [日志主题与格式](../examples/04_日志主题与格式.ipynb)

`core.py` 定义公共接口，`themes.py` 定义预设，`formatting.py` 负责终端/文件表示。非有限浮点数字段按 `nan/inf/-inf` 字符串写入 JSONL，避免生成非标准 JSON。

Loguru 的独立实例适配集中在 `_backend.py`；升级 Loguru 时需运行日志隔离和来源定位测试。仓库使用方式见 [仓库概览](../README.md)，本地开发记录统一放在根目录 `PROJECT.md`。
