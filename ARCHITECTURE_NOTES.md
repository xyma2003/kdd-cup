# KDD Cup 2026 DataAgent-Bench 参赛经验沉淀

> 本文档记录 xyma 分支的整体架构、关键设计决策、工程技巧和 prompt 工程经验。
> 最终得分：v5 Qwen3.5-35B 32.55 / 50 题。

---

## 一、任务概述

比赛给定自然语言问题 + context/ 目录（CSV / JSON / SQLite / 文档），要求 Agent 自主分析并输出 `prediction.csv`（结果表格）。评测用 F1 近似指标对比 gold.csv。

核心挑战：数据格式多样（结构化 + 半结构化 + 纯文本），问题语义歧义，跨文件 JOIN，大文档处理。

---

## 二、总体架构

```
question + context/
       │
       ▼
  [classifier]          ← 纯规则，零 LLM，按文件类型路由
       │
   ┌───┴──────────────────────────┐
   │             │                │
TYPE_SQL   TYPE_PANDAS      TYPE_HYBRID / TYPE_DOC
   │             │                │
sql_solver  pandas_solver   structured_solver
   │             │                │
DuckDB跨源    subprocess       状态机4步
执行           代码沙箱       EXTRACT → BUILD → EXECUTE
```

### 2.1 分类器（classifier.py）

纯规则路由，零 LLM 开销：
- `TYPE_SQL`：有 `.db` / `.sqlite` 文件 → sql_solver
- `TYPE_PANDAS`：纯 CSV/JSON → pandas_solver
- `TYPE_HYBRID`：文档 + 结构化数据 → structured_solver
- `TYPE_DOC`：纯文档 → structured_solver（无结构化数据兜底）

**设计原则**：路由不消耗 LLM token，且失败代价低（可以在 solver 内降级）。

### 2.2 难度预算（DifficultyBudget）

同样是纯规则估计，基于可观测信号打分：
- 文档大小（越大越难）
- 跨源文件数（需要 JOIN 时更难）
- DB 总列数（schema 越复杂越难）
- 问题关键词（"ratio", "percentage", "for each" 等）

按得分分 4 档：easy（跳过意图校验，修复 1 次）→ extreme（不跳过，修复 2 次）。

**价值**：不同难度分配不同 LLM 预算，easy 题省钱，hard 题多给机会。

---

## 三、sql_solver：3 步意图分离架构

### 设计思路

传统做法是直接"schema + question → SQL"。问题：LLM 在看到 schema 后容易被表名/列名带偏，选错聚合方式或 GROUP BY 逻辑。

**改进**：先理解语义意图，再生成 SQL，两步之间加一个 reviewer 纠错。

```
Step 1: intent_extractor (不看 schema)
  question → {aggregation, group_by, output_cols, filters, requires_distinct, ...}

Step 1.5: intent_reviewer (只改 op 运算符，其余不动)
  intent → intent_corrected

Step 2: sql_generator (intent + schema → SQL)
  把 intent 作为强约束注入 prompt
```

### 关键 Prompt 设计

**意图提取 prompt**（`_INTENT_SYSTEM_PROMPT`）只做一件事：结构化理解问题语义，不产生 SQL。关键字段：
- `aggregation`：AVG / SUM / COUNT / MAX / MIN / null
- `group_by`：有则表示需要分组，null 表示全局聚合（单行结果）
- `requires_distinct`：是否需要 DISTINCT
- `output_cols`：结果列数（约束 LLM 不多选列）

**意图校验 prompt** 的核心原则：只修 op 运算符（`>`/`>=`/`<`/`<=` 是最常见的 off-by-one 错误），绝不动其他字段。"If unsure, do NOT change anything."

**SQL 生成 prompt** 把 intent 作为 "verified — follow this exactly" 强约束，将 `group_by` 的存在与否翻译成 CRITICAL 规则动态注入。

### 执行层

用 DuckDB 实现跨源联合查询（SQLite + CSV + JSON 注册为视图，同一 SQL 可 JOIN）。失败时降级到逐个 SQLite 尝试。

### 修复流程

```
执行成功 + 非空 → verify_and_fix（语义复查）
                  → is_correct=true → 返回
                  → is_correct=false → 带诊断重新生成 SQL（最多2轮）

执行失败/空结果 → 规则修复（fuzzy match 列名/表名）
              → 还失败 → LLM 修复（max_repair_rounds 轮）
```

修复 prompt 针对不同错误类型（空结果 vs 重复行）提供专门的诊断提示，引导 LLM 往正确方向改。

**ID 列清理**（`_drop_id_columns`）：当 intent 明确指定了列数，且实际列数多出，自动丢弃明显是主键的列（`_id$`, `^id$` 等）。

---

## 四、pandas_solver：代码沙箱执行

### 设计

LLM 生成完整 pandas 代码，用 `subprocess` 在隔离进程执行。结果通过 `print("__RESULT__" + df.to_json(...))` 传回，解析 stdout 获取 DataFrame。

**好处**：代码执行与主进程隔离，崩溃不影响主流程。任何 import 错误 / 内存溢出都被 subprocess 边界隔离。

### 大文件保护

pandas 不适合全量加载大文件。实现了快速行数估算（读头部 4KB + 文件大小估算），超过 5 万行自动降级到 sql_solver（DuckDB 流式读取）。

```python
# 只读 4KB 估算，不全量加载
file_size = path.stat().st_size
head = f.read(4096)
avg_line_bytes = len(head) / head.count(b"\n")
estimated_rows = int(file_size / avg_line_bytes)
```

### Join 推断

自动扫描所有 DataFrame 的列名，找共享列名提示 LLM 可能的 JOIN key，减少 LLM 猜测。

### Schema 枚举值展示

TEXT 列且 unique 值 ≤ 30 时，把所有实际值展示给 LLM（"use EXACT spelling in filters"）。这一条直接解决了大量 case mismatch 问题（LLM 自己猜的字符串经常大小写不对）。

---

## 五、structured_solver：状态机 4 步 pipeline

专门处理"文档 + 结构化数据"的混合任务，核心问题：文档里有实体名称/ID，结构化数据里有属性，需要先从文档提取映射再做查询。

### 状态机

```
EXTRACT_ENTITIES → BUILD_QUERY → EXECUTE → DONE
                                   ↓ (失败)
                              BUILD_QUERY (最多2次重试)
                                   ↓ (仍失败)
                               FAILED
```

### Layer 1：Python 正则提取（零 LLM）

优先用正则精准提取 rec-ID 映射，覆盖 5 种格式：
1. `NAME (Registry ID: recXXX)` 
2. `corrected to NAME (recXXX)`
3. `recXXX ... is NAME`
4. `(recXXX)` 前面的名词短语
5. 分子编号 `TR391`、`Case ID N` 等非 rec 格式

**关键细节**：用 "最后一个状态" 覆盖 "initially X, corrected to Y" 的纠错情况。真实 rec_id 必须含数字（排除 "reconstructive" 等英文单词误匹配）。

### Layer 2：LLM 补充提取（正则失败时）

- 小文档（< 12000 字符）：全文一次 LLM 调用
- 大文档（≥ 12000 字符）：BM25 关键词打分筛选相关段落，单次 LLM 调用

替代了早期版本的"盲目分块 + 并行60次 LLM"方案（task_408 超时问题的根源）。

### BM25 大文档筛选

```
切块（1200字符，150字符重叠）
→ 每块对问题关键词打分（词级重叠率）
→ 取 top-8 块，按原文位置排序
→ 合并间距 < 2000 字符的相邻块（保留"前陈述后纠正"的上下文）
→ 截断到 14000 字符上限
```

带重叠的切块保证跨块边界的信息不会丢失。合并相邻块保证"先说 X，后纠正为 Y"的情况不被切断。

### 代码执行预注入

LLM 生成的代码运行在预注入了以下变量的环境里：
- `csv_paths`：文件名 → 绝对路径 dict（LLM 用 `pd.read_csv(csv_paths['x.csv'])`）
- `db_paths`：同上，SQLite 连接用
- `{table}_df`：JSON 文件预加载为 DataFrame（直接用，不用 json.load）
- `id_to_name`：已提取的 ID→名称映射
- `id_field`：链接列名提示

这样 LLM 不需要关心文件路径，只需要写业务逻辑。

---

## 六、knowledge.md 双层解析

knowledge.md 是比赛提供的数据字典，包含字段定义、阈值、SQL 示例、别名。

### Layer 1：正则提取

- SQL 代码块 → few-shot 示例
- `**FieldName:** description` → 字段定义
- 阈值模式（`values above N`、`BETWEEN X AND Y`、`>= N`）→ 正常值范围
- 别名模式 → entity aliases

### Layer 2：LLM 语义补充

正则无法理解自然语言描述的阈值（如"WBC 正常范围为 3.5-9.0"）。每道题额外调用一次 LLM，提取正则没覆盖的字段定义和阈值，与正则结果取并集（正则优先，LLM 只补缺）。

**价值**：让 SQL/pandas 生成时有更完整的业务语义上下文，减少 LLM 猜测枚举值。

---

## 七、Qwen3 Thinking Mode 接入

Qwen3 on SiliconFlow 的 thinking mode 实现有个特殊行为：最终答案也在 `reasoning_content` 里，`content` 字段为空。

**适配方案**（`_extract_final_answer_from_thinking`）：
1. 找最后一个代码块（SQL/JSON/代码）
2. 找 "Output exactly `X`" / "final answer:" 模式
3. 找最后一个非 bullet、非标题的段落
4. 兜底：返回末尾 200 字符

这个 3 层 fallback 覆盖了 Qwen3 各种输出格式变体。

**收益**：thinking mode 下模型先推理再回答，在需要理解语义的复杂题（COUNT DISTINCT、嵌套 JSON、多表 JOIN）上提升显著（v4→v5 新增 8 分）。

---

## 八、5 层容错 JSON 解析

LLM 输出经常不规范，需要强健的解析器：

```
Level 1: ```json ... ``` 代码块
Level 2: 任意 ``` ... ``` 代码块
Level 3: json.loads 整个响应（裸 JSON）
Level 4: 括号匹配法找最外层 { }（比 rfind 准确，不被字符串里的 } 干扰）
Level 5: 正则逐字段提取关键字段（专为 intent JSON 兜底设计）
```

Level 4 的括号匹配法处理了 Level 3 无法处理的"JSON 前后有说明文字"情况。Level 5 处理了 JSON 结构破损但关键字段可用的情况。

**原则**：解析失败时保守返回 `is_correct: true`（不破坏已有结果），而不是抛出异常导致任务失败。

---

## 九、API 重试策略

```python
# 429 TPM 指数退避（最多5次）
for attempt in range(5):
    try:
        response = client.chat.completions.create(...)
        break
    except APITimeoutError:
        if attempt < 1: sleep(10); continue
        raise  # 超时2次直接失败
    except APIError as exc:
        if exc.status_code == 429:
            sleep(60 * (attempt + 1))  # 60/120/180/240s
        else:
            raise
```

timeout 设 600s（thinking mode 单次可能需要 30-180s）。

---

## 十、失分根因总结（50 题公开集）

| 根因类型 | 典型题目 | 分析 |
|---------|---------|------|
| 表/列选择错误 | task_163（查 expense 表而非 budget.json） | LLM 在多表环境下倾向于选有明确 schema 的表，忽略隐含数据源 |
| 字段语义歧义 | task_80（qualifying.number vs drivers.number） | 同名字段在不同表有不同语义，需要 knowledge 明确说明 |
| 空结果未修复 | task_173（跨 DB+JSON JOIN） | DuckDB 跨源 JOIN 路径问题，规则修复覆盖不到 |
| 多余行/列 | task_199（filter 条件太宽）、task_379（TALLY 带 count 列） | intent.output_cols 约束有时被 LLM 忽略 |
| verify 误判 | task_200（单行聚合被 verify 改坏） | verify 对"1行结果"的判断在某些情况下错误触发 |
| 随机波动 | task_11、task_415 | thinking mode 下输出非确定性，小样本下波动大 |
| 数据缺失/架构限制 | task_396（字段不存在）、task_352（金额在散文里） | 无法修复，是数据集的天花板 |

---

## 十一、可复用的 Prompt 规则库

以下是从失分案例中归纳的、值得在未来项目中复用的 SQL/pandas prompt 规则：

### 聚合类
- "average monthly" → `SUM(col) / COUNT(DISTINCT month)`，不是 `AVG(col)`
- "单个数值" → 只返回聚合结果，不返回明细行
- "entity with highest/lowest metric" → 只返回 entity 列，不返回 metric 列

### 比较运算符
- "more than N" → `> N`（不是 `>= N`）
- "at least N" → `>= N`
- 意图校验层单独处理这条，防止生成层遗漏

### 并列值（Tied Rows）
- "lowest/highest X" 如果有多行并列，要返回所有并列行
- SQL：`WHERE col = (SELECT MIN(col) FROM ...)`，不用 `LIMIT 1`
- pandas：`df[df['col'] == df['col'].min()]`，不用 `.nsmallest(1)`

### 结果格式
- 不返回 id/record_id/primary key 列，除非问题明确要求
- full name = 两列（first_name + last_name），不拼接
- 除法前过滤分母为 0/NULL

### JOIN 去重
- 当一个实体对应多行时，JOIN 后用 CTE 先 `SELECT DISTINCT entity_id`，再关联
- 否则会产生重复行

### 枚举值
- 把枚举列的所有实际值展示给 LLM，标注"use EXACT spelling"
- 大小写不匹配是最常见的空结果原因之一

---

## 十二、架构演进轨迹

| 版本 | 模型 | 得分 | 关键改进 |
|------|------|------|---------|
| v2 | DeepSeek-V4-Flash | 33.00 | 难度估计 + LLM 兜底 + SQL 修复 |
| v4 | Qwen3.5-35B | 27.40 | 代码清理（切换模型退步） |
| v5 | Qwen3.5-35B | **32.55** | Thinking mode + verify + BM25 大文档 + knowledge 双层解析 |

从 v4→v5 的核心经验：**模型切换要重新评估所有假设**。v4 的 structured_solver 列数限制在 DeepSeek 下没问题，在 Qwen 下变成了瓶颈。thinking mode 需要专门处理 reasoning_content，不能直接用标准 API。

---

## 十三、未来可以继续优化的方向

1. **verify iron rule**：`group_by == null && aggregation != null` → 单行聚合，直接跳过 verify（防止 task_200 类误判）
2. **TIED ROWS 规则**：pandas/SQL 里统一加，返回所有并列最值行
3. **TALLY 结果格式**：只返回被统计的实体列，不返回 count 列
4. **drivers.number 语义**：在 Formula_1 数据集的 prompt 里说明 `drivers.number` 是永久车手编号，优先于其他表的 number 列
5. **budget.json 优先级**：当问题涉及费用/预算时，JSON 文件应该优先于 expense CSV

---

*记录时间：2026-05-19*
