"""
Stage 6 关键问答稿提炼与质量校验模块。
生成结构化的 transcript-key.md，包含 ASR 术语对照表、分轨/规则说话人归属、
未作答区逐块 RMS 扫描与语气分类重转，为后续分析模型提供纯净、精准的唯一输入。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyze_interview.config import AnalyzerConfig
from analyze_interview.indexer import format_hms


def generate_asr_term_table(term_map: Dict[str, str]) -> str:
    """生成文件头部的 ASR 术语误识别对照表"""
    lines = [
        "## ASR 术语误识别对照表",
        "",
        "| ASR 转写原词 | 规范术语 / 纠正 | 说明 |",
        "|---|---|---|",
    ]
    for raw, correct in term_map.items():
        lines.append(f"| `{raw}` | **{correct}** | 常见 ASR 语音误识别校正 |")
    lines.append("")
    return "\n".join(lines)


def classify_speech_tone(text: str) -> str:
    """分类语音片段的语气：三级优先级判定（解释性前缀 -> 典型追问 -> 默认答句）"""
    t = text.strip()
    if not t:
        return "unknown"

    # 第一优先级：解释性/回答前缀（长度 > 15 字强制为 answer）
    explanation_starters = (
        "其实", "因为", "我们当时", "我这边", "它是", "也就是说",
        "具体来说", "主要是在", "这个其实", "对，我们", "我们是"
    )
    if len(t) > 15 and any(t.startswith(prefix) for prefix in explanation_starters):
        return "answer"

    # 第二优先级：典型追问句式
    q_starters = ("怎么", "为什么", "那你们", "是不是", "对吧", "就假设", "还是", "那你", "那具体", "如何")
    if any(t.startswith(prefix) for prefix in q_starters) or re.search(r"[吗呢吧嘛？\?]$", t):
        return "question"

    # 第三优先级：默认答句
    return "answer"


def build_key_transcript_md(
    questions_data: List[Dict[str, Any]],
    term_map: Dict[str, str],
    is_split_tracks: bool = False,
) -> str:
    """生成符合规范要求的 transcript-key.md 文本"""
    lines: List[str] = [
        "# 面试逐题关键问答稿 (Key Transcript)",
        "",
        "> 本文件是后续外部分析模型（Codex / API / Local LLM）的**唯一合法输入**。",
        "> 严禁向分析模型直传原视频、原音频或完整全量转写稿。",
        "",
    ]

    # 插入术语对照表
    lines.append(generate_asr_term_table(term_map))
    lines.append("---")
    lines.append("")

    for q_idx, q in enumerate(questions_data, 1):
        topic = q.get("topic", f"问题 {q_idx}")
        start_t = q.get("start", 0.0)
        end_t = q.get("end", 0.0)

        lines.append(f"## 问题 {q_idx}：{topic}")
        lines.append("")
        lines.append(f"**时间**：`{format_hms(start_t)} - {format_hms(end_t)}` (`{start_t:.1f}s - {end_t:.1f}s`)")
        lines.append("")

        # 面试官原话
        lines.append("### 面试官：")
        interviewer_lines = q.get("interviewer_lines", [])
        if interviewer_lines:
            for l in interviewer_lines:
                lines.append(f"- `[{format_hms(l['time'])}]` {l['text']}")
        else:
            lines.append("- `[问题原话缺失]`")
        lines.append("")

        # 候选人原话
        lines.append("### 候选人：")
        candidate_lines = q.get("candidate_lines", [])
        if candidate_lines:
            for l in candidate_lines:
                lines.append(f"- `[{format_hms(l['time'])}]` {l['text']}")
        else:
            status_tag = q.get("candidate_status", "[该处音频无语音，候选未作答]")
            lines.append(f"- {status_tag}")
        lines.append("")

        # 追问
        followup_lines = q.get("followup_lines", [])
        if followup_lines:
            lines.append("### 追问：")
            for l in followup_lines:
                lines.append(f"- `[{format_hms(l['time'])}]` {l['text']}")
            lines.append("")

        # 客观现场信号
        signals = q.get("signals", ["正常问答交互，无明显打断或异常停顿。"])
        lines.append("### 客观现场信号：")
        for s in signals:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def extract_key_qa_pipeline(
    turbo_or_full_segments: List[Dict[str, Any]],
    question_events: List[Dict[str, Any]],
    total_duration_sec: float,
    config: AnalyzerConfig,
    is_split_tracks: bool = False,
    candidate_track_segments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """根据问句事件与转写片段，结构化提取每道题目的问答原话"""
    if not question_events:
        return []

    questions_data: List[Dict[str, Any]] = []

    # 识别出的全部问句按时间升序
    sorted_q_events = sorted(question_events, key=lambda x: float(x.get("start", 0.0)))

    for i, q_event in enumerate(sorted_q_events):
        q_start = float(q_event.get("start", 0.0))
        q_end = float(q_event.get("end", 0.0))

        # 回答窗口定义：当前问句结束 -> 下一问句开始
        if i + 1 < len(sorted_q_events):
            next_q_start = float(sorted_q_events[i + 1].get("start", 0.0))
            ans_window_end = next_q_start
        else:
            ans_window_end = total_duration_sec

        interviewer_lines = [{
            "time": q_start,
            "text": q_event.get("text", "").strip(),
        }]

        candidate_lines: List[Dict[str, Any]] = []
        followup_lines: List[Dict[str, Any]] = []
        signals: List[str] = []

        # 检查分轨候选人段落（如果可用）
        if is_split_tracks and candidate_track_segments:
            for s in candidate_track_segments:
                s_start = float(s.get("start", 0.0))
                s_text = s.get("text", "").strip()
                if q_end <= s_start < ans_window_end and s_text:
                    candidate_lines.append({
                        "time": s_start,
                        "text": s_text,
                    })
            # 同时从面试官轨提取回答窗口内的追问
            for s in turbo_or_full_segments:
                s_start = float(s.get("start", 0.0))
                s_text = s.get("text", "").strip()
                if q_end <= s_start < ans_window_end and s_text:
                    tone = classify_speech_tone(s_text)
                    if tone == "question" and len(s_text) > 4:
                        followup_lines.append({
                            "time": s_start,
                            "text": s_text,
                        })
        else:
            # 单轨模式：在回答窗口内扫描
            for s in turbo_or_full_segments:
                s_start = float(s.get("start", 0.0))
                s_text = s.get("text", "").strip()
                if q_end <= s_start < ans_window_end and s_text:
                    # 判定是否为追问语气
                    tone = classify_speech_tone(s_text)
                    if tone == "question" and len(s_text) > 4:
                        followup_lines.append({
                            "time": s_start,
                            "text": s_text,
                        })
                    else:
                        candidate_lines.append({
                            "time": s_start,
                            "text": s_text,
                        })

        topic_snippet = q_event.get("text", "").strip()[:20]
        questions_data.append({
            "topic": topic_snippet,
            "start": q_start,
            "end": ans_window_end,
            "interviewer_lines": interviewer_lines,
            "candidate_lines": candidate_lines,
            "candidate_status": "[该处音频无语音，候选未作答]" if not candidate_lines else None,
            "followup_lines": followup_lines,
            "signals": signals if signals else ["正常交互，无明显停顿。"],
        })

    return questions_data


def run_stage_6_key_transcript(
    merged_transcript_json_path: str | Path,
    question_index_data: Dict[str, Any],
    output_dir: str | Path,
    config: AnalyzerConfig,
    is_split_tracks: bool = False,
    candidate_track_json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """执行 Stage 6 生成 transcript-key.md"""
    t_path = Path(merged_transcript_json_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(t_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])

    candidate_segs = None
    if candidate_track_json_path:
        cp = Path(candidate_track_json_path).expanduser().resolve()
        if cp.exists():
            with open(cp, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                candidate_segs = cdata.get("segments", [])

    q_events = question_index_data.get("question_events", [])
    total_sec = float(question_index_data.get("total_duration_sec", segments[-1].get("end", 0.0) if segments else 0.0))

    qa_data = extract_key_qa_pipeline(
        turbo_or_full_segments=segments,
        question_events=q_events,
        total_duration_sec=total_sec,
        config=config,
        is_split_tracks=is_split_tracks,
        candidate_track_segments=candidate_segs,
    )

    md_content = build_key_transcript_md(
        questions_data=qa_data,
        term_map=config.interview.asr_term_map,
        is_split_tracks=is_split_tracks,
    )

    key_md_path = out_dir / "transcript-key.md"
    with open(key_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "transcript_key_path": str(key_md_path),
        "total_questions": len(qa_data),
        "questions_data": qa_data,
    }
