# Nsight System 使用
## 0. Nsight System 的功能
## 1. Nsight System 的安装
todo
## 2. 使用方法1：命令行
## 3. 使用方法2：GUI
## 4. NVTX标记
## 5. 具体代码实现
### 1. NVTX标记的实现的三种方法：
todo
1. 使用 torch.cuda.nvtx
   PyTorch 提供了对 NVTX 的原生支持，通过 torch.cuda.nvtx 模块可以直接插入标记。
```python
import torch

# 标记一个代码段开始
torch.cuda.nvtx.range_push("Forward Pass")

output = model(input)
loss = criterion(output, target)

# 结束标记
torch.cuda.nvtx.range_pop()
```
此外，还可以使用上下文管理器（推荐，更安全）
PyTorch 提供了 range 上下文管理器，自动处理 push 和 pop：
```python
import torch

with torch.cuda.nvtx.range("Forward Pass"):
    output = model(input)
    loss = criterion(output, target)

with torch.cuda.nvtx.range("Backward Pass"):
    loss.backward()

with torch.cuda.nvtx.range("Optimizer Step"):
    optimizer.step()
``` 
2. 创建装饰器函数用于标记模型函数
可以定义一个装饰器，自动为函数添加 NVTX 标记：
```
def nvtx_range(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = f"{func.__module__}.{func.__qualname__}"
        with torch.cuda.nvtx.range(name):
            return func(*args, **kwargs)
    return wrapper
```

3. 嵌套标记（支持层级分析）
嵌套标记适合分析复杂流程：
```
with torch.cuda.nvtx.range("Training Epoch 0"):
    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.cuda.nvtx.range(f"Batch {batch_idx}"):
            with torch.cuda.nvtx.range("Data to GPU"):
                data, target = data.cuda(), target.cuda()

            with torch.cuda.nvtx.range("Forward"):
                output = model(data)
                loss = criterion(output, target)

            with torch.cuda.nvtx.range("Backward"):
                loss.backward()

            with torch.cuda.nvtx.range("Optimizer"):
                optimizer.step()
                optimizer.zero_grad()
```

## 参数设置


## 📁 一、输出与文件控制

| 参数 | 说明 |
|------|------|
| `-o, --output=<filename>` | 指定输出文件名前缀（如 `report.nsys-rep`）。支持以下占位符：<br>• `%n`：自动编号<br>• `%p`：进程 PID<br>• `%h`：主机名<br>• `%q{ENV}`：环境变量值<br>• `%%`：表示 `%` 字符<br>默认为 `report%n`。 |
| `--auto-report-name=true/false` | 是否根据应用信息自动生成报告名，格式为：<br>`[进程名][GPU型号][分辨率][图形API] 时间戳.nsys-rep`<br>默认 `false`。 |
| `-f, --force-overwrite=true/false` | 是否覆盖已存在的同名文件（包括 `.qdstrm`, `.nsys-rep` 等）。<br>默认 `false`。 |
| `--export=<format>[,<format>...]` | 额外导出数据格式：<br>可选：`none`, `sqlite`, `hdf`, `text`, `json`, `arrow`, `arrowdir`, `parquetdir`<br>例如：`--export=sqlite,json`<br>默认 `none`。 |

---

## ⏱️ 二、采集时间与触发方式

| 参数 | 说明 |
|------|------|
| `-d, --duration=<seconds>` | 采集持续时间（秒）。<br>默认 `0`（无限，直到应用退出或手动停止）。 |
| `-y, --delay=<seconds>` | 延迟开始采集的时间（秒）。<br>默认 `0`。 |
| `-Y, --start-later=true/false` | 延迟采集，直到通过 `nsys start` 命令手动启动。<br>优先级高于 `--delay`，默认 `false`。 |
| `--stop-on-exit=true/false` | 是否在目标程序退出时自动停止采集。<br>若设为 `false`，则必须设置 `--duration > 0`。<br>默认 `true`。 |
| `--kill=true/false` | 采集结束后是否终止目标程序。<br>默认 `true`。 |

---

## 🔘 三、采集范围控制（Start/Stop 触发）

| 参数 | 说明 |
|------|------|
| `-c, --capture-range=<mode>` | 指定采集启动方式：<br>• `none`：立即开始（默认）<br>• `cudaProfilerApi`：调用 `cudaProfilerStart()` 开始<br>• `nvtx`：进入某个 NVTX 范围时开始<br>• `hotkey`：按下热键开始（仅图形应用） |
| `--capture-range-end=<behavior>` | 范围结束后的处理方式：<br>• `none`：忽略<br>• `stop`：停止采集，程序继续<br>• `stop-shutdown`：停止并关闭会话（默认）<br>• `repeat[:N]`：重复采集 N 次<br>• `repeat-shutdown:N`：重复 N 次后关闭 |
| `-p, --nvtx-capture=<range>[@domain]` | 配合 `--capture-range=nvtx` 使用，指定触发的 NVTX 标记：<br>例：`--nvtx-capture=MyRange@MyDomain` |
| `--hotkey-capture=F1~F12` | 设置热键触发采集（仅 Windows，F10 不支持）。<br>默认 `F12`。 |

---

## 🎯 四、API 跟踪（Trace）

| 参数 | 说明 |
|------|------|
| `-t, --trace=<apis>` | 指定要跟踪的 API，多个用逗号分隔：<br>• `cuda`：CUDA API 调用<br>• `cuda-hw`：CUDA 硬件活动（内核执行）<br>• `nvtx`：NVIDIA Tools Extension 标记<br>• `cublas`, `cudnn`, `cusolver`, `cusparse`：对应库 API<br>• `opengl`, `vulkan`, `dx11`, `dx12`：图形 API<br>• `python-gil`：Python GIL 切换<br>• `none`：不跟踪任何 API<br>**默认：`cuda,nvtx,opengl`** |

> ⚠️ 注意：`cuda-hw` 是采集 GPU 内核执行时间的关键！

---

## 🖥️ 五、GPU 性能指标（Metrics）——关键！

> 💡 **这是你问题的核心：没有 GPU 指标是因为没启用这些参数！**

| 参数 | 说明 |
|------|------|
| `--gpu-metrics-devices=<list>` | 指定采集 GPU 指标的设备：<br>• `none`（默认）<br>• `cuda-visible`：当前可见的 CUDA 设备<br>• `all`：所有 GPU<br>• 或指定 GPU ID（如 `0,1`）<br>可用 `--gpu-metrics-devices=help` 查看设备列表 |
| `--gpu-metrics-frequency=<Hz>` | GPU 指标采样频率（Hz）：<br>范围：10 ~ 200,000 Hz<br>默认 `10,000`（每 0.1ms 一次） |
| `--gpu-metrics-set=<set>` | 指定采集的指标集合：<br>可用 `--gpu-metrics-set=help` 查看支持的集合（如 `basic`, `memory`, `sm`, `tensor` 等）<br>默认选择第一个兼容所有设备的集合 |

> ✅ **推荐开启 GPU 指标采集的组合：**
```bash
--gpu-metrics-devices=cuda-visible \
--gpu-metrics-frequency=10000 \
--gpu-metrics-set=basic,memory,sm,tensor
```

---

## 🧮 六、CUDA 特性增强

| 参数 | 说明 |
|------|------|
| `--cuda-trace-all-apis=true/false` | 是否跟踪所有 CUDA API（包括低频调用），可能增加开销。<br>默认 `false`。 |
| `--cuda-memory-usage=true/false` | 跟踪 GPU 内存使用情况（占用高）。<br>默认 `false`。 |
| `--cuda-event-trace=auto/true/false` | 跟踪 CUDA Event 设备端完成时间，提升同步分析精度。<br>需 CUDA 12.8+，默认 `false`。 |
| `--cuda-graph-trace=graph/node[:host-only/host-and-device]` | 控制 CUDA Graph 跟踪粒度：<br>• `graph`：整体跟踪（低开销）<br>• `node`：节点级跟踪（高开销）<br>• `host-and-device`：支持设备端启动（12.3+） |
| `--cuda-flush-interval=<ms>` | 设置 CUDA 数据缓冲区刷新间隔（ms），0 表示填满即刷。<br>默认 `0`。 |

---

## 🖼️ 七、图形 API 专用设置

| 参数 | 说明 |
|------|------|
| `--opengl-gpu-workload=true/false` | 是否跟踪 OpenGL 的 GPU 工作负载。<br>默认 `true`。 |
| `--vulkan-gpu-workload=individual/batch/none` | Vulkan 工作负载跟踪方式：<br>• `individual`：逐个命令缓冲区<br>• `batch`：按 `vkQueueSubmit` 批量<br>• `none`：不跟踪<br>默认 `individual`。 |
| `--dx12-gpu-workload=individual/batch/none` | DX12 工作负载跟踪方式，同上。<br>默认 `individual`。 |

---

## 🧠 八、采样与上下文切换

| 参数 | 说明 |
|------|------|
| `-s, --sample=process-tree/system-wide/none` | CPU 采样范围：<br>• `process-tree`：目标进程及其子进程（默认）<br>• `system-wide`：全系统（需管理员权限）<br>• `none`：关闭采样 |
| `--sampling-frequency=<Hz>` | CPU 采样频率（100 ~ 8000 Hz），默认 `1000` Hz。 |
| `--cpuctxsw=process-tree/system-wide/none` | 跟踪 CPU 上下文切换：<br>需管理员权限，`process-tree` 默认。 |
| `--gpuctxsw=true/false` | 跟踪 GPU 上下文切换（需 CUDA 驱动 r435.17+）。<br>默认 `false`。 |
| `--isr=true/false` | 跟踪中断服务程序（ISR）和 DPC（仅 Windows，需管理员权限）。<br>默认 `false`。 |

---

## 🐍 九、Python 支持

| 参数 | 说明 |
|------|------|
| `--python-sampling=true/false` | 是否对 Python 线程进行采样（获取调用栈）。<br>默认 `false`。 |
| `--python-sampling-frequency=<Hz>` | Python 采样频率（1 ~ 2000 Hz），默认 `1000` Hz。 |
| `--python-functions-trace=<json_file>` | 指定 JSON 文件，标记要跟踪的 Python 函数。<br>提供模板路径供参考。 |
| `--pytorch=autograd-nvtx,functions-trace,...` | PyTorch 专用模式：<br>• `autograd-nvtx`：自动插入 autograd NVTX 标记<br>• `functions-trace`：加载预定义函数跟踪<br>• 可组合使用，如 `autograd-nvtx,functions-trace` |
| `--dask=functions-trace/none` | Dask 专用支持，重命名线程并启用函数跟踪。 |

---

## 🏷️ 十、NVTX 控制

| 参数 | 说明 |
|------|------|
| `--nvtx-domain-include=<domain1,domain2>` | 仅包含指定 NVTX 域的标记。 |
| `--nvtx-domain-exclude=<domain1,domain2>` | 排除指定 NVTX 域的标记。<br>⚠️ `include` 和 `exclude` 互斥。 |

---

## 🛠️ 十一、高级与调试选项

| 参数 | 说明 |
|------|------|
| `--stats=true/false` | 采集结束后生成统计摘要（SQLite 数据库）。<br>**必须开启才能在 GUI 中看到“Summary”页和指标汇总！**<br>默认 `false`。 |
| `--resolve-symbols=true/false` | 是否解析符号（函数名）。Windows 默认 `false`。 |
| `--debug-symbols=<path1:path2>` | 指定符号文件目录（Linux/Windows），用 `:` 分隔。 |
| `--command-file=<file>` | 从文件读取参数，命令行参数优先级更高。 |
| `--env-var=A=B,C=D` | 设置目标程序的环境变量。 |
| `--inherit-environment=true/false` | 是否继承当前环境变量。<br>默认 `true`。 |
| `-w, --show-output=true/false` | 是否将目标程序的 stdout/stderr 输出到控制台。<br>默认 `true`。 |
| `--wait=primary/all` | 等待目标进程还是整个进程树结束。<br>默认 `all`。 |

---

## 🧩 十二、实验性功能（Experimental）

| 参数 | 说明 |
|------|------|
| `--enable=<plugin>[,arg1,arg2...]` | 启用插件（实验性）。<br>用 `--enable=help` 查看可用插件。 |

---

## ❓ 如何查看帮助子集？

```bash
nsys profile --help=tag
```

支持的标签（tags）：
- `cuda`, `nvtx`, `gpu`, `python`, `trace`, `stats`, `export`, `capture`, `hotkey`, `windows` 等

例如：
```bash
nsys profile --help=gpu        # 查看 GPU 相关参数
nsys profile --help=cuda       # 查看 CUDA 参数
nsys profile --help=stats      # 查看统计相关
```

---

## ✅ 解决你问题的关键总结

### ❓ 为什么 GUI 打开 `.nsys-rep` 没有 GPU 指标？

因为你**没有启用 GPU 指标采集**！

### ✅ 正确做法：

```bash
nsys profile ^
  --output=my_report ^
  --trace=cuda,cuda-hw,nvtx ^
  --gpu-metrics-devices=cuda-visible ^
  --gpu-metrics-set=basic,memory,sm,tensor ^
  --gpu-metrics-frequency=10000 ^
  --stats=true ^
  python train.py
```

### ✅ 生成命令行报告：

```bash
nsys stats my_report.nsys-rep
```

或导出 CSV：

```bash
nsys stats ^
  --report gpukernsum,gputrace ^
  --output-format=csv ^
  --output=results ^
  my_report.nsys-rep
```

---

## 📌 建议流程

1. **先测试基本采集**：
   ```bash
   nsys profile -o test --trace=cuda,cuda-hw ./app
   ```

2. **再加入 GPU 指标**：
   ```bash
   --gpu-metrics-devices=cuda-visible --gpu-metrics-set=basic,memory --stats=true
   ```

3. **最后用 `nsys stats` 生成报告**。

