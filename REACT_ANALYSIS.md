# ReAct Agent — 架构分析

> 分支：react  
> 基于：官方基线 ReAct + zing 分支最佳实践  

> ⚠️ **架构说明**：ReAct 是 `react` 分支引入的**实验性架构**，目前未在 production 调用（`cli.py` 硬编码 `use_xyma=True`，production 实际跑 `solvers/` 规则分类器架构）。本文件分析 ReAct 的设计思路与潜力，作为后续优化的参考。要切换到 ReAct，需将 `cli.py:144,237` 的 `use_xyma=True` 改为 `False`。

---

## 架构概述

ReAct（Reasoning + Acting）是一种让 LLM 在"思考"和"行动"之间交替迭代的 Agent 框架。每一步模型输出 `thought → action → observation`，循环直到调用 `answer` 工具提交答案。

```
用户问题
    ↓
[thought] 我需要先看数据结构
[action] list_context / execute_python / read_csv ...
[observation] 工具返回结果
    ↓
[thought] 基于观察，我需要...
[action] execute_python (写 SQL 或 pandas 代码)
[observation] 执行结果
    ↓
[action] answer → 提交最终答案
```

---

## 亮点与优势

### 1. 通用性强
不需要预先对任务分类，模型自己决定怎么探索数据。无论是 SQL、CSV、散文文档还是混合数据，都走同一套循环。相比专用 solver 架构，对未见题型的适应能力更强。

### 2. 上下文预加载（新增）
在任务开始前自动注入：
- 所有文件列表和大小
- CSV 列名
- SQLite DB 表结构和字段类型
- JSON 文件结构（table name、列名、记录数）
- Doc 文件预览（前400字符）
- knowledge.md 完整内容（最多8000字符）

这让模型在第一步就有完整的数据地图，减少无效的探索步骤。

### 3. 详细的 Prompt 规则（来自 zing）
系统 prompt 包含：
- **列裁剪规则**：只返回问题明确要求的列，多余列扣分
- **Tied rows 规则**：最低/最高值用 `WHERE value = MIN(value)` 不用 `LIMIT 1`
- **精度规则**：不 ROUND，保持完整浮点精度
- **DOC 解析规则**：用 regex 从散文文档提取结构化数据
- **7 个 WRONG/CORRECT 示例**：对比展示常见错误和正确做法

### 4. 格式容错重试
JSON 解析失败时自动注入格式纠正 prompt 重试（最多2次），避免格式错误直接导致步骤失败。

### 5. 卡死检测与反思
- 同一个 action 重复3次 → 触发 reflection，提示换思路
- 使用了 70% 步数仍无进展 → 触发 reflection
- 避免模型在死路上无限循环

### 6. 强制提交保底
- 最后2步自动注入 force answer prompt
- token 预算超限时强制提交
- 连续5次错误后强制提交
- 确保不会因为超步数而完全没有答案

### 7. 历史压缩
超过8步后只保留最近3步，防止 context 无限增长导致 OOM 或费用爆炸。

---

## 缺点

### 1. 准确率天然低于专用 Solver
ReAct 每步都依赖 LLM 决策，错误可以在任意步骤引入并累积。专用 Solver 的 SQL/pandas 生成是一次性的，失败了有明确的错误信息可以修复；ReAct 的中间错误更难追踪。

### 2. Token 消耗大
每道题需要多轮对话，每次都带上完整历史。30步上限下，一道复杂题可能消耗 5-10 倍于专用 solver 的 token。

### 3. 随机性更高
多步推理积累了更多的随机性，同一道题两次结果差异比专用 solver 更大。temperature=0 只控制最终采样，不控制每一步的推理路径。

### 4. 工具调用在 thinking mode 下不稳定
Qwen3 开启 thinking mode 时，结构化的 JSON 工具调用格式容易被推理过程干扰，导致解析失败率上升。这是 ReAct 架构在 thinking mode 下的已知问题。

### 5. 历史压缩有信息损失
超过8步后压缩历史，之前的探索结果会丢失，可能导致模型重复已经做过的操作。

---

## 可优化方向

### 短期（1-2天）
1. **关闭 thinking mode**：ReAct 不适合 thinking mode，关闭后工具调用更稳定，每步更快
2. **增加 execute_python 优先级提示**：在 prompt 里明确"优先用 execute_python，不要反复 list_context"
3. **工具输出截断优化**：大结果自动截断，防止 observation 撑爆 context

### 中期（1周）
4. **动态步数分配**：easy 题给10步，hard 题给30步，节省 token
5. **错误分类处理**：区分"代码执行错误"和"结果语义错误"，给出更有针对性的修复提示
6. **knowledge.md 结构化注入**：把 knowledge.md 里的 SQL 示例、字段定义单独抽出来，放在更显眼的位置

### 长期
7. **混合架构**：用分类器先判断任务类型，简单的 SQL/pandas 题走专用 solver，只有文档混合型题才走 ReAct
8. **工具扩展**：增加 DuckDB 跨源查询工具，让 ReAct 也能一条 SQL 查 SQLite + CSV + JSON

---

## 天花板分析

**理论上限**：如果模型足够强，ReAct 应该能解决所有题——它不预设任何假设，完全靠推理。对于专用 solver 无法处理的题（散文文档里有答案、跨格式数据、非标准 ID 映射），ReAct 反而有优势。

**实际天花板**（Qwen3.5-35B，30步）：
- 纯 SQL/CSV 题：~65-70%（和专用 solver 相当，但 token 消耗3倍）
- HYBRID 混合题：~50-60%（可能优于专用 solver，因为更灵活）
- 散文文档题：~30-40%（专用 solver 基本为0）

**整体预估**：公开 demo 集约 28-32 分（50分满分），和专用 solver 持平，但在 HYBRID/DOC 类题上可能略有优势，在纯 SQL 类题上略有劣势。

---

## 与 xyma 专用 Solver 对比

| 维度 | ReAct (react 分支) | 专用 Solver (main 分支) |
|------|-------------------|----------------------|
| **通用性** | 高，无预设假设 | 低，依赖分类器路由 |
| **纯 SQL 准确率** | 中 | 高 |
| **文档型题准确率** | 中 | 低（架构限制） |
| **Token 消耗** | 高（多轮对话） | 低（1-3次 LLM） |
| **随机性** | 高 | 中 |
| **可调试性** | 难（中间步骤多） | 易（明确的 solver 路径） |
| **Thinking mode 兼容** | 差（工具调用不稳定） | 好（无工具调用） |
| **代码复杂度** | 低（通用循环） | 高（多个专用 solver） |
