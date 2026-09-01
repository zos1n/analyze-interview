"""
Stage 7 逐题深度复盘与分析模块。
支持 codex 提示词生成 (codex-analysis-prompt.md)、API 直连复盘与本地模式，
严格遵循 7 维度评估框架与事实/推断/无法判断三分法。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

CODEX_PROMPT_TEMPLATE = """# 面试录屏深度逐题复盘分析指令

你是一位资深的 AI 前端与大模型应用架构专家面试官。
请基于下方提供的 **面试逐题关键问答稿 (transcript-key.md)**，对整场面试进行深度、客观、严谨的逐题复盘评估。

---

## 核心原则与硬性约束

1. **输入限制**：仅基于下方提供的 key transcript 进行评估，不得臆造未发生的对话细节。
2. **三分法原则**：所有结论必须严格区分：
   - **【事实】**：直接引述原话并附带时间戳。
   - **【推断】**：基于回答内容与技术逻辑推导，注明推断依据。
   - **【无法判断】**：信息不足时明确指出缺少何种证据，禁止臆测。
3. **禁止臆断**：
   - 没追问 ≠ 答得好（可能是面试官放弃深入或时间紧迫）。
   - 候选人主观感受 ≠ 面试官评价。
   - 单个题目失误 ≠ 最终淘汰唯一原因。
   - 禁止在无客观信号时猜测面试官心理态度。

---

## 每题固定的 7 维度评估结构

针对 transcript-key.md 中的每一道题目，依次输出：

1. **【考察目标】**：面试官提出该题的核心考察维度（深度、广度、工程落地能力、架构思维等）。
2. **【回答覆盖内容】**：候选人回答中覆盖了哪些核心技术点与业务场景。
3. **【缺失内容与风险】**：回答中遗漏的关键机制、技术漏洞或潜在风险。
4. **【客观现场信号】**：打断、长停顿、反复确认、快速跳题等可观察现场事实。
5. **【失分判定】**：明确划分为以下三类之一：
   - `[明确失分]`：技术原理错误、关键机制答错或暴露明显短板。
   - `[隐性失分]`：回答泛泛、缺乏量化指标、答非所问或未体现核心竞争力。
   - `[无明显问题]`：回答完整、逻辑清晰、有技术深度与落地支撑。
6. **【更好的回答结构】**：结合 STAR 原则与高阶架构认知，提供更优的回答要点与叙事逻辑。
7. **【对结果影响与置信度】**：该题在整体技术面评级中的权重影响及置信度（高/中/低）。

---

## 最终总结（Summary）

在逐题分析完成后，提供：
1. **核心优势与技术亮点 Top 3**
2. **核心失分项与风险清单 Top 3**
3. **后续面试定向提升建议（含话术与项目叙事建议）**

---

## 待分析的关键问答稿内容

```markdown
{transcript_key_content}
```
"""


def generate_codex_analysis_prompt(
    transcript_key_path: str | Path,
    output_prompt_path: str | Path,
) -> str:
    """生成用于 Codex / 外部模型的分析提示词文件 (codex-analysis-prompt.md)"""
    tk_p = Path(transcript_key_path).expanduser().resolve()
    out_p = Path(output_prompt_path).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(tk_p, "r", encoding="utf-8") as f:
        tk_content = f.read()

    prompt = CODEX_PROMPT_TEMPLATE.format(transcript_key_content=tk_content)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(prompt)

    return str(out_p)


def generate_review_skeleton(
    questions_data: List[Dict[str, Any]],
    output_review_path: str | Path,
) -> str:
    """生成初始的 interview-review.md 模板骨架"""
    out_p = Path(output_review_path).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 面试录屏逐题复盘评估报告 (Interview Review)",
        "",
        "> 本报告基于全本地转写提炼的 `transcript-key.md` 进行结构化复盘。",
        "> 遵循事实/推断/无法判断三分法与 7 维度分析框架。",
        "",
        "---",
        "",
    ]

    for idx, q in enumerate(questions_data, 1):
        topic = q.get("topic", f"问题 {idx}")
        start_t = q.get("start", 0.0)
        end_t = q.get("end", 0.0)

        lines.extend([
            f"## 问题 {idx}：{topic}",
            "",
            f"- **时间戳**：`{start_t:.1f}s - {end_t:.1f}s`",
            "- **失分判定**：`[待评估]`",
            "",
            "### 1. 【考察目标】",
            "【推断】待分析...",
            "",
            "### 2. 【回答覆盖内容】",
            "【事实】待提取...",
            "",
            "### 3. 【缺失内容与风险】",
            "【推断】待评估...",
            "",
            "### 4. 【客观现场信号】",
            "【事实】待记录...",
            "",
            "### 5. 【失分判定】",
            "- 判定：`[明确失分 / 隐性失分 / 无明显问题]`",
            "- 依据：【推断】待说明...",
            "",
            "### 6. 【更好的回答结构】",
            "建议结构：",
            "1. ...",
            "",
            "### 7. 【对结果影响与置信度】",
            "- 影响程度：中等",
            "- 置信度：高",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 综合复盘总结",
        "",
        "### 核心优势亮点",
        "1. ...",
        "",
        "### 核心改进建议",
        "1. ...",
        "",
    ])

    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(out_p)
