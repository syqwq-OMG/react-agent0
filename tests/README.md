# 作业测试路线

测试依据仓库根目录 `react-agent-assignment.pdf` v4.0 的正文契约；附录 A 的章节编号与正文有出入时，以正文签名和异常约定为准。这是一套作业验收测试，不包含核心组件的参考实现。

在根目录运行：

```bash
uv sync --python 3.11 --extra dev
uv run --extra dev python -m pytest --collect-only -q
uv run --extra dev python -m pytest tests/test_protocol.py -q
```

一次只完成一个模块，再运行对应测试。`adapters.py` 使用延迟导入，所以没有实现后续章节不会妨碍收集和运行前面章节的测试。待实现入口会报 `NotImplementedError: Implement react_agent.<module>.<function> first`；不要把它改成跳过测试或固定返回测试答案。

| PDF 正文 | 测试文件 | adapter 转发的模块 | 核心检查 |
|---|---|---|---|
| §3 协议 | `test_protocol.py` | `protocol.py` | 单个 JSON、字段类型、note、可省略 evidence、非法证据和输入 |
| §4 检索 | `test_retrieval.py` | `retrieval.py` | 正则词集合、正分筛选、同分标题排序、query/k/title 边界、原句不变 |
| §5 工具 | `test_tools.py` | `tools.py` | 注入工具、参数错误、异常、JSON 结果、可配置截断、跨题隔离 |
| §6 消息 | `test_prompting.py` | `prompting.py` | system/user、协议和预算、新历史列表、Observation 前缀 |
| §6 循环 | `test_loop.py` | `agent.py` | 两跳 London、连续观察、精确计次、预算、失败恢复和完整 trace |
| §7 后端 | `test_providers.py` | `providers.py` | 两个后端、显式 URL/模型/超时、正文提取、空响应、认证和超时异常 |
| §8 日志 | `test_tracing.py` | `tracing.py` | UTF-8 JSONL、run ID 隔离、残缺记录与重复记录诊断 |
| §9 数据 | `test_dataset.py` | `dataset.py` | 必填字段、重复 ID/标题、数据类型、运行数据与 gold 分离 |
| §10 指标 | `test_metrics.py` | `metrics.py` | 官方归一化、Counter、yes/no/noanswer、空值、证据集合 |
| §10 评测 | `test_evaluation.py` | `evaluation.py` | SHA256 切分及跨进程复现、预测全覆盖、失败保留、产物和续跑 |
| 测试框架 | `test_adapters.py` | 无需实现模块 | 20 个入口的参数、默认值与返回值转发 |

后端测试拦截 HTTPX transport，既适用于 OpenAI SDK，也适用于直接使用 HTTPX 的客户端。它不会连接真实服务，不需要 API Key。所有单元测试默认禁止 socket 连接，避免调试时意外访问远端；真实后端连通性和真实实验应由独立脚本运行，并写入报告。

测试检查公开行为，不要求特定 prompt 原文、内部类名或私有辅助函数。JSONL 损坏或重复数据可以抛出有说明的 `ValueError`，也可以明确警告/记录诊断并修复或跳过；不得静默将残缺记录当作完成。工具中断信号 `KeyboardInterrupt` 必须向外传播。

批量测试使用临时目录中的两道人工样本，模拟成功、后端失败、重复运行与中途中断。它检查规定的五类运行产物、完整 ID 覆盖和实际调用次数；模型工厂测试替身可接受任意位置/关键字参数，不强制学生实现选择某一种工厂调用方式。PDF 没有固定 aggregate 的键名，所以测试不凭空指定汇总字典结构。

完整检查：

```bash
uv run --extra dev python -m pytest tests/test_adapters.py -q
uv run --extra dev python -m pytest -q
```

单元测试不能替代书面回答、真实数据摘要、后端真实调用、官方 scorer 交叉核验、对照/消融与失败案例分析；这些仍需按 PDF 提交。当前仓库没有核心实现，完整作业测试失败是预期状态。
