下面是你贴出的 `ncu --help` 的**中文文档版**。我保留了原有的结构与参数名称，仅将说明文字翻译为中文，尽量逐条对等，不做省略或简化（英文字段/标志保持不变以便对照命令行）。

# 用法

```
usage: ncu.exe [options] [program] [program-arguments]
```

# 通用选项（General Options）

```
  -h [ --help ]                         打印此帮助信息。
  -v [ --version ]                      打印版本号。
  --mode arg (=launch-and-attach)       选择与目标应用交互的方式：
                                          launch-and-attach
                                          （启动并附加以进行分析）
                                          launch
                                          （启动并挂起，等待稍后附加）
                                          attach
                                          （附加到已启动的应用）
  -p [ --port ] arg (=49152)            与目标应用连接的基础端口。
  --max-connections arg (=64)           连接到目标应用的最大端口数量。
  --config-file arg (=1)                使用 config.ncu-cfg 配置文件设置参数。默认在当前工作目录
                                        和 "%APPDATA%\NVIDIA Corporation" 目录中查找。
  --config-file-path arg                覆盖配置文件的默认路径。
```

# 启动选项（Launch Options）

```
  --check-exit-code arg (=1)            检查应用退出码；若不为 0 则报错。
                                        若设置此项，--replay-mode application 在第一轮若退出码非 0
                                        会停止。
  --injection-path-32 arg (=../windows-desktop-win7-x86)
                                        覆盖 32 位注入库的默认路径。
  --injection-path-64 arg               覆盖 64 位注入库的默认路径。
  --preload-library arg                 在注入库前预加载指定共享库。
  --call-stack                          启用 CPU 调用栈采集。默认采集 native 类型，
                                        参见 --call-stack-type。
  --call-stack-type arg                 设置要采集的调用栈类型（可指定多个）。隐式启用 --call-stack。
                                          native (default)
                                          （采集常规 CPU 调用栈）
                                          python
                                          （采集 Python 调用栈）
  --nvtx                                启用 NVTX 支持。
  --target-processes arg (=all)         选择需要分析的进程：
                                          application-only
                                          （仅分析应用进程）
                                          all
                                          （分析应用及其子进程）
  --target-processes-filter arg         以逗号分隔的表达式，筛选需要分析的进程：
                                          <process name> 精确匹配要包含的进程名。
                                          regex:<expression> 使用正则匹配要包含的进程名。
                                            对于将正则字符视为特殊字符的 shell，需要用引号转义。
                                          exclude:<process name> 精确匹配要排除的进程名。
                                        仅匹配可执行文件名部分。过滤按首个匹配停止。
                                        一旦指定了任意“包含”过滤器，则只分析被包含筛中的进程。
```

# 附加选项（Attach Options）

```
  --hostname arg                        设置连接目标的主机名 / IP 地址。
```

# 通用分析选项（Common Profile Options）

```
  --kill arg (=0)                       当已采集到 --launch-count 指定的数量后，终止目标应用。
  --replay-mode arg (=kernel)           多次回放 kernel 启动以收集所需数据的机制：
                                          kernel (default)
                                          （在应用执行期间，透明地回放单个 kernel 启动。）
                                          application
                                          （多次重新启动整个应用。需应用执行具有确定性。）
                                          range
                                          （在应用执行期间，回放 NVTX 范围的 kernel / API 调用。）
                                          app-range
                                          （通过多次重启应用来分析范围（不捕获 API）。需确定性。）
  --app-replay-buffer arg (=file)       应用回放缓冲的存放位置：
                                          file (default)
                                          （回放数据缓存在临时文件；分析完成后生成报告。）
                                          memory
                                          （回放数据缓存在内存；分析过程中生成报告。）
  --app-replay-match arg (=grid)        应用回放时的 kernel 匹配策略（按进程与设备）：
                                          name
                                          （按名称匹配）
                                          grid (default)
                                          （按名称与网格/块尺寸匹配）
                                          all
                                          （按名称、网格/块尺寸、上下文 ID、流 ID 匹配）
  --app-replay-mode arg (=balanced)     应用回放时的 kernel 匹配模式：
                                          strict   : 所有回放轮次必须按完全相同顺序匹配。
                                          balanced : 所有回放轮次需都匹配，
                                          (default)  但无需严格顺序一致。
                                          relaxed  : 仅对成功在回放轮次间匹配到的 kernel 生成结果；
                                                     未匹配的 kernel 将被丢弃。
  --graph-profiling arg (=node)         CUDA Graph 分析模式：
                                          node (default)
                                          （逐个分析图中的 kernel 节点）
                                          graph
                                          （对整张图进行分析）
  --range-replay-options arg            范围回放选项，以逗号分隔：
                                          enable-greedy-sync
                                          （在捕获期间，为适用的延迟 API 插入上下文同步）
                                          disable-host-restore
                                          （禁用恢复由设备写入的主机分配）
  --list-sets                           列出搜索路径中找到的所有“指标集合（section sets）”。
  --set arg                             指定要采集的“指标集合”标识。如果未指定，采集 basic 集合。
  --list-sections                       列出搜索路径中找到的所有“面板（sections）”。
  --section-folder arg                  指定 section 文件的搜索路径（不递归）。
  --section-folder-recursive arg        指定 section 文件的搜索路径（递归）。
  --section-folder-restore              将官方自带的 section 文件恢复到默认目录，
                                        或恢复到由 --section-folder 指定的目录。
  --list-rules                          列出搜索路径中的所有分析规则（rules）。
  --apply-rules arg (=1)                对采集到的 sections 应用分析规则。
                                        若未设置 --rule，则应用所有可用规则。可选值：
                                          on/off
                                          yes/no
  --rule arg                            指定要应用的规则标识。启用后等同于 --apply-rules yes。
  --import-source arg (=0)              若编译时带 -lineinfo，则将关联的 CUDA 源文件永久导入报告。
                                        可选值：on/off, yes/no。
  --source-folders arg                  以逗号分隔的搜索路径；在启用 --import-source 时用于查找
                                        并导入关联 CUDA 源文件（递归）。
  --list-metrics                        列出基于所选 sections 将要收集的所有指标。
  --query-metrics                       查询系统上设备可用的指标（可配合 --devices / --chips 过滤）。
                                        默认该选项返回的指标名需要后缀才可被采集。详见
                                        --query-metrics-mode。
  --query-metrics-mode arg (=base)      设置查询指标的模式（隐式启用 --query-metrics）。可选：
                                          base (default)
                                          （只显示基础指标名）
                                          suffix
                                          （显示带后缀的指标名。此模式下通过 --metrics 指定
                                           要展开查询的基础指标。）
                                          all
                                          （显示完整指标名）
  --query-metrics-collection arg (=profiling)
                                        设置查询哪一类指标集合（隐式启用 --query-metrics）。可选：
                                          device
                                          （查询 CUDA 设备属性）
                                          groups
                                          （查询指标分组）
                                          launch
                                          （查询启动属性）
                                          numa
                                          （查询 NUMA 拓扑指标）
                                          nvlink
                                          （查询与 NVLink 相关指标）
                                          pmsampling
                                          （查询 PM 采样可用指标）
                                          profiling (default)
                                          （查询可用于分析的指标）
                                          source
                                          （查询源码级指标）
                                          stats
                                          （查询由分析器生成的统计类指标）
                                          warpsampling
                                          （查询可用于周期性采样 warp 程序计数器与调度状态的指标）
  --list-chips                          列出可与 --chips 搭配的所有受支持芯片。
  --chips arg                           指定要查询指标的芯片（以逗号分隔）。
  --profile-from-start arg (=1)         设置是否从应用启动起就开始分析。可选：on/off, yes/no。
  --disable-profiler-start-stop         禁用开始/停止分析的控制。设置后，cu(da)ProfilerStart/Stop
                                        API 将被忽略。
  --quiet                               抑制所有分析器输出。
  --verbose                             打印更详细的分析器输出。
  --cache-control arg (=all)            控制分析期间 GPU 缓存的行为。可选：
                                          all
                                          none
  --clock-control arg (=base)           控制分析期间 GPU 时钟的行为。可选：
                                          base
                                          （将 GPU 时钟锁定到基准频率）
                                          none
                                          （不锁定时钟）
                                          reset
                                          （重置 GPU 时钟并退出）
  --pipeline-boost-state arg (=stable)  控制 Tensor Core Boost 状态。为获得可预测的
                                        运行间性能，建议设置为稳定模式。
                                          stable
                                          （将 Tensor Core Boost 设为稳定）
                                          dynamic
                                          （将 Tensor Core Boost 设为动态）
```

# 启动过滤选项（Filter Profile Options）

```
  --devices arg                         指定要启用分析的设备（以逗号分隔）。默认启用所有设备。
  --filter-mode arg (=global)           设置对 kernel 启动应用过滤器的模式。可选：
                                           global (default) : 将提供的启动过滤器“整体”应用。
                                           per-gpu          : 分别在每块 GPU 上应用过滤器。
                                                              （此模式下 --launch-count / --launch-skip 有效）
                                           per-launch-config: 按每个 GPU 启动参数（网格、块大小、
                                                              共享内存）分别应用 kernel 与启动过滤器。

  --kernel-id arg                       指定匹配 kernel 的标识，格式：
                                        "context-id:stream-id:[name-operator:]kernel-name:invocation-nr"。
                                        可跳过某些字段，如 "::foobar:2" 表示匹配任意上下文/流中的
                                        第 2 次名为 "foobar" 的调用。
                                        也可用 ":7:regex:^foo:" 匹配流 7 中以 "foo" 开头的 kernel
                                        （解释基于 --kernel-name-base）。
  -k [ --kernel-name ] arg              以如下方式过滤 kernel：
                                          <kernel name> 精确匹配内核名。
                                          regex:<expression> 使用正则匹配内核名。
  --kernel-name-base arg (=function)    设置 --kernel-name、--kernel-id 和显示的 kernel 名称的基准：
                                          function
                                          demangled
                                          mangled
  --rename-kernels arg (=1)             对解符后的 kernel 名称进行简化/重命名。
                                        使用配置文件进行重命名。默认在当前工作目录与
                                        "%APPDATA%\NVIDIA Corporation" 中查找 ncu-kernel-renames.yaml。
                                        可用 --rename-kernels-export 将简化后的名称导出到配置文件。
  --rename-kernels-export arg (=0)      将已重命名/简化的 kernel 名导出到配置文件（默认导出到
                                        "%APPDATA%\NVIDIA Corporation"）。
  --rename-kernels-path arg             覆盖 kernel 重命名配置文件路径。仅在重命名/导出时有效。
  -c [ --launch-count ] arg             限制要收集的启动次数。仅对匹配过滤条件的启动计数。
  -s [ --launch-skip ] arg (=0)         在开始分析前跳过若干次 kernel 启动。仅对匹配的启动计数。
  --launch-skip-before-match arg (=0)   在开始分析前跳过若干次启动。对所有启动计数（不论是否匹配）。
  --section arg                         通过以下方式之一收集合适的“面板（section）”：
                                          <section identifier> 精确匹配 section 标识。
                                          regex:<expression> 用正则匹配 section 标识。
                                        若未指定该选项，则收集默认的 section set。
                                        无法收集的 section 指标通常只会产生警告。
  --metrics arg                         指定要分析的所有指标（以逗号分隔）。
                                        该选项支持以下前缀：
                                          regex:<expression> 展开为所有“部分匹配表达式”的指标。
                                                             若需全匹配，请用 ^...$。
                                          group:<name>       展开为该“指标组”的所有指标。
                                                             组名参见 section 文件。
                                          breakdown:<metric> 展开为该高层吞吐指标的输入明细项。
                                                             若该指标不支持分解，则不添加。
                                        若某指标需要后缀才有效，且未使用前缀，本选项会自动
                                        展开为所有可用的一级子指标。
                                        不能采集的指标会报错。
  --disable-extra-suffixes              禁用额外后缀（avg、min、max、sum）的采集。仅采集显式指定的。
  --nvtx-include arg                    向 NVTX 过滤器添加“包含”语句，允许基于 NVTX 范围选择要分析的内核。
  --nvtx-exclude arg                    向 NVTX 过滤器添加“排除”语句，允许基于 NVTX 范围排除内核。
  --range-filter arg                    筛选匹配的 NVTX 范围或由 cu(da)ProfilerStart/Stop 创建的
                                        起止范围中的特定实例。格式：
                                        <yes/no/on/off>:<start/stop range instance(s)>:<NVTX range instance(s)>
                                           <yes/no/on/off> : 默认为 'no/off'。若设为 'yes/on'，
                                                             则在每个 start/stop 范围内，NVTX
                                                             范围编号从 1 开始。
                                           可用正则式区间表示编号，如 [2-4] 或 2|3|4，表示分析
                                           第 2/3/4 个匹配范围实例。
                                           NVTX 范围编号将基于 --nvtx-include 指定的匹配范围计数。
  --nvtx-push-pop-scope arg (=thread)   指定 push/pop 范围的作用域：
                                           thread (default) : NVTX Push/Pop 范围按线程计数。
                                           process          : 按进程计数。
  --native-include arg                  向“原生 CPU 调用栈过滤器”添加包含语句，基于原生栈帧
                                        选择要分析的内核。
  --native-exclude arg                  向“原生 CPU 调用栈过滤器”添加排除语句。
  --python-include arg                  向“Python CPU 调用栈过滤器”添加包含语句，基于 Python 栈帧筛选。
  --python-exclude arg                  向“Python CPU 调用栈过滤器”添加排除语句。
```

# PM 采样选项（PM Sampling Options）

```
  --pm-sampling-interval arg (=0)       设置 PM 采样间隔（单位为周期或纳秒，取决于架构），
                                        为 0 时自动决定。
  --pm-sampling-buffer-size arg (=0)    设置设备侧 PM 采样缓冲区大小（字节），为 0 时自动决定。
  --pm-sampling-max-passes arg (=0)     设置 PM 采样的最大 pass 数，为 0 时自动决定。
```

# Warp 状态采样选项（Warp State Sampling Options）

```
  --warp-sampling-interval arg (=auto)  设置 warp 状态采样周期，范围 [0..31]。
                                        实际频率为 2^(5+value) 个周期。设为 'auto' 时由分析器自动确定
                                        尽可能高的采样频率，同时避免跳样或缓冲溢出。
  --warp-sampling-max-passes arg (=5)   设置 warp 状态采样的最大 pass 数。
  --warp-sampling-buffer-size arg (=33554432)
                                        设置设备侧保存 warp 状态样本的缓冲大小（字节）。
```

# 文件选项（File Options）

```
  --log-file arg                        将所有工具输出发送到指定文件或标准通道。
                                          文件会被覆盖；若不存在将创建新文件。
                                          若文件名为 "stdout" 则输出到标准输出（默认）。
                                          若文件名为 "stderr" 则输出到标准错误。
  -o [ --export ] arg                   设置结果报告的输出文件名。若未设置，使用临时文件（完成后删除）。
  -f [ --force-overwrite ]              强制覆盖所有输出/section/配置文件（若存在则覆盖）。
  -i [ --import ] arg                   设置读取已有报告结果的输入文件。
  --open-in-ui                          分析完成后在图形界面中打开报告，而不是在终端打印。
```

# 控制台输出选项（Console Output Options）

```
  --csv                                 以 CSV 形式输出。默认同时隐式设置 --print-units base。
  --page arg (=details)                 选择要输出的报告页：
                                          details: sections 与规则
                                          raw:     所有收集到的原始指标
                                          source:  源代码
                                          session: 会话与设备属性
  --print-source arg                    选择源码视图类型：
                                          sass
                                          ptx
                                          cuda
                                          cuda,sass
                                         SASS 与 cuda,sass 视图支持与指标相关联的源码高亮。
                                         当使用 --metrics / --section 指定了指标/面板时，相关性可用。
                                         建议限制指标数量，以便表格在一行内显示。
  --resolve-source-file arg             指定逗号分隔的文件路径列表，用于解析源码视图中的文件引用。
  --print-details arg (=header)         选择在 details 页面输出 section 的哪一部分：
                                          header (default)
                                          （输出 section 表头中的所有指标）
                                          body
                                          （输出 section 正文中的所有指标）
                                          all
                                          （输出该 section 的全部指标）
  --print-metric-name arg (=label)      设置“指标名称”列显示方式：
                                          label (default)  显示指标标签
                                          name             显示指标内部名
                                          label-name       同时显示标签与内部名
  --print-units arg (=auto)             设置单位显示方式：
                                          auto (default)   自动按合适量级缩放
                                          base             使用基础单位显示
  --print-metric-attribution            在“绿色上下文（Green Context）”结果中显示指标归属级别。
  --print-fp                            将所有数值型指标以浮点数形式显示。
  --print-kernel-base arg (=demangled)  设置 kernel 名称的显示基准。选项同 --kernel-name-base。
  --print-metric-instances arg (=none)  设置多实例指标的输出模式：
                                          none (default)   仅显示汇总值
                                          values           显示汇总值 + 所有实例值
                                          details          显示汇总值 + 关联 ID + 实例值
  --print-nvtx-rename arg (=none)       选择 NVTX 用于重命名的方式：
                                          none (default)   不使用 NVTX 重命名
                                          kernel           以最近的 NVTX push/pop 范围重命名 kernel
  --print-rule-details                  打印规则结果的附加细节，如表格与关键性能指标。
                                        （CSV 模式下当前无效）
  --print-summary arg (=none)           设置摘要输出模式：
                                          none
                                          per-gpu
                                          per-kernel
                                          per-nvtx.
```

# 用法示例（Usage Patterns）

```
使用 --mode 选择工具用法：
  启动并分析一个 CUDA 应用：
      ncu.exe CuVectorAdd

  启动应用以便稍后附加：
      ncu.exe --mode=launch MyApp
  附加到已启动的应用：
      ncu.exe --mode=attach --hostname 127.0.0.1
  也可使用图形界面启动或附加到应用。

选择特定的 kernel 启动进行分析：
  分析名为 'foo' 的 kernel 的前两次启动：
      ncu.exe -k foo -c 2 CuVectorAdd

载入已有的报告：
      ncu.exe --import myReport

--nvtx-include 与 --nvtx-exclude 的用法：
  ncu.exe --nvtx --nvtx-include "Domain A@Range A"
     仅分析位于“Domain A”的“Range A”起止范围内的内核。
  ncu.exe --nvtx --nvtx-exclude "Range A]"
    分析除默认域栈顶上“Range A” push/pop 范围内的内核以外的所有内核。
  ncu.exe --nvtx --nvtx-include "Range A" --nvtx-exclude "Range B"
     仅分析“Range A”范围内、但不在“Range B”范围内的内核（默认域）。

--native-include / --python-include 与 --native-exclude / --python-exclude 的用法：
  ncu.exe --native-include "Module A@File A@Function A" --native-exclude "Module B@File B@Function A"
    分析通过模块 A / 文件 A / 函数 A 启动的内核，但排除模块 B / 文件 B / 函数 A 启动的内核。
  ncu.exe --python-include "Module A@File A@Function A" --python-exclude "Module B@File B@Function A"
    分析通过文件 A / 函数 A（Python 栈）启动的内核，但排除文件 B / 函数 A 启动的内核。
  ncu.exe --native-include "Module A@@[Function A"
    分析位于原生 CPU 调用栈底部的“模块 A 的函数 A”所启动的内核。
  ncu.exe --python-exclude "File A@Function A]"
    分析除位于 Python 调用栈顶端的“文件 A 的函数 A”所启动的内核以外的所有内核。
  注意：Module 与 File 名为可选。若未提供，将匹配所有模块与文件。
        对二者均可指定完整路径或仅指定名称。
```

——以上为 `ncu --help` 的中文对照文档。需要我把它导出成 Markdown/PDF 或加上目录与书签，告诉我你想要的格式即可。
****