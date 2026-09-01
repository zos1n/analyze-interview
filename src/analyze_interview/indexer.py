"""
Stage 4 本地问题索引与密度聚类模块。
实现确定性主题簇聚类（ASCII 词边界 + 中文子串匹配）、扩展问句模式识别与回答窗口重定义，
生成三张视图并计算覆盖率以进行自动精转决策。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analyze_interview.config import AnalyzerConfig

# 常见短回应词与排除词
SHORT_REPLIES = {"嗯", "对", "好", "ok", "OK", "对对对", "大概理解", "行", "好的", "没问题", "是的"}
EXPLANATORY_STARTERS = ("对", "因为", "其实", "就是说", "它是", "这个", "我们", "我这边")


def format_hms(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 或 MM:SS"""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def match_keywords(
    text: str,
    ai_coding_kws: List[str],
    project_kws: List[str],
) -> List[str]:
    """对文本进行关键词匹配（ASCII 词按词边界匹配，中文按子串匹配）"""
    hits: List[str] = []
    text_clean = text.strip()
    if not text_clean:
        return hits

    all_kws = [("AI_Coding", kw) for kw in ai_coding_kws] + [("Project", kw) for kw in project_kws]

    for category, kw in all_kws:
        if kw.isascii() and kw.isalnum():
            # ASCII 词使用单词边界 + 不区分大小写
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text_clean, re.IGNORECASE):
                hits.append(kw)
        else:
            # 中文或混合词使用子串包含
            if kw.lower() in text_clean.lower():
                hits.append(kw)

    return list(dict.fromkeys(hits))  # 保序去重


def is_question_candidate(
    text: str,
    question_patterns: List[str],
    rms: Optional[float] = None,
    silence_threshold: float = 0.005,
) -> Tuple[bool, Optional[str]]:
    """扩展问句语法判定（结合句末疑问助词、疑问代词、选择句式与反问引导，并应用排除规则）"""
    t = text.strip()
    if not t:
        return False, None

    # 排除规则 1：静音段
    if rms is not None and rms < silence_threshold:
        return False, None

    # 排除规则 2：短回应词 (<= 2 词)
    if t in SHORT_REPLIES or (len(t) <= 3 and any(r in t for r in ["嗯", "对", "好", "OK", "ok"])):
        return False, None

    # 排除规则 3：解释性段落（除非开头即为疑问词）
    if any(t.startswith(prefix) for prefix in EXPLANATORY_STARTERS):
        # 如果不是紧随疑问词，则排除
        is_direct_q = any(t.startswith(prefix + q) for prefix in ["", "那", ""] for q in ["为什么", "怎么", "如何"])
        if not is_direct_q:
            return False, None

    # 规则 A: 显式匹配 question_patterns (为什么/怎么/如何/有没有/介绍一下/具体说说)
    for p in question_patterns:
        if p in t:
            return True, f"模式:{p}"

    # 规则 B: 句末疑问助词 (吗|呢|吧|嘛|？|?)
    if re.search(r"[吗呢吧嘛？\?][\s\.\,\，\。]*$", t):
        return True, "句末疑问词"

    # 规则 C: 疑问代词/副词 (哪|什么|啥|谁|为什么|怎么|如何|为啥|多少|多久)
    q_pronouns = ["哪", "什么", "啥", "谁", "为什么", "怎么", "如何", "为啥", "多少", "多久"]
    for qp in q_pronouns:
        if qp in t:
            return True, f"疑问词:{qp}"

    # 规则 D: 确认/选择句式 (是吧|对吧|是吗|是不是|有没有|还是|或者说)
    q_choices = ["是吧", "对吧", "是吗", "是不是", "有没有", "还是", "或者说"]
    for qc in q_choices:
        if qc in t:
            return True, f"选择/确认句:{qc}"

    # 规则 E: 反问/连续追问引导 (以 那为什么|那你们|那怎么|那你 开头)
    if re.match(r"^(那为什么|那你们|那怎么|那你|那具体)", t):
        return True, "追问引导词"

    return False, None


def find_must_include_intervals(
    segments: List[Dict[str, Any]],
    total_duration_sec: float,
    reverse_patterns: List[str],
    silence_threshold: float = 0.005,
) -> List[Dict[str, Any]]:
    """计算规范要求的必须包含区间：
    1. 面试开头（首个非静音语音起 ±2 分钟）
    2. 反问环节（结尾 1/3 区域搜索 reverse_patterns，或后 8 分钟兜底）
    3. 结束前 5 分钟
    """
    must_intervals: List[Dict[str, Any]] = []

    # 1. 开头
    first_speech = 0.0
    for s in segments:
        rms = s.get("_rms", 1.0)
        if rms >= silence_threshold and len(s.get("text", "").strip()) > 1:
            first_speech = float(s.get("start", 0.0))
            break
    start_a = max(0.0, first_speech - 5.0)
    start_b = min(total_duration_sec, first_speech + 120.0)
    must_intervals.append({
        "name": "面试开头",
        "start": start_a,
        "end": start_b,
        "reason": "必须包含区间: 面试开头",
    })

    # 2. 反问环节（文末 1/3 区域）
    last_third_start_idx = int(len(segments) * 2 / 3)
    last_third_segs = segments[last_third_start_idx:] if segments else []
    reverse_hits = []
    for s in last_third_segs:
        text = s.get("text", "")
        if any(rp in text for rp in reverse_patterns):
            reverse_hits.append(float(s.get("start", 0.0)))

    if reverse_hits:
        first_rev = min(reverse_hits)
        rev_a = max(0.0, first_rev - 90.0)
        rev_b = min(total_duration_sec, first_rev + 180.0)
        must_intervals.append({
            "name": "反问环节",
            "start": rev_a,
            "end": rev_b,
            "reason": "必须包含区间: 反问环节",
        })
    else:
        # 兜底后 8 分钟
        rev_a = max(0.0, total_duration_sec - 480.0)
        must_intervals.append({
            "name": "反问环节 (兜底后8分钟)",
            "start": rev_a,
            "end": total_duration_sec,
            "reason": "必须包含区间: 反问环节 (未检测到明确表达兜底)",
        })

    # 3. 结束前 5 分钟
    end_a = max(0.0, total_duration_sec - 300.0)
    must_intervals.append({
        "name": "结束前5分钟",
        "start": end_a,
        "end": total_duration_sec,
        "reason": "必须包含区间: 结束前5分钟",
    })

    return must_intervals


def cluster_segments_by_density(
    hit_segments: List[Dict[str, Any]],
    max_gap_sec: float = 180.0,
) -> List[Dict[str, Any]]:
    """密度聚类：命中段按间隔 < 180s 归为一簇（规范阶段 4，不做 ±90s 扩展）"""
    if not hit_segments:
        return []

    sorted_hits = sorted(hit_segments, key=lambda x: float(x.get("start", 0.0)))
    clusters: List[Dict[str, Any]] = []

    curr_cluster_segs = [sorted_hits[0]]
    for seg in sorted_hits[1:]:
        prev_end = float(curr_cluster_segs[-1].get("end", 0.0))
        curr_start = float(seg.get("start", 0.0))

        if curr_start - prev_end <= max_gap_sec:
            curr_cluster_segs.append(seg)
        else:
            # 闭合当前簇
            c_start = float(curr_cluster_segs[0].get("start", 0.0))
            c_end = float(curr_cluster_segs[-1].get("end", 0.0))
            kws = list(dict.fromkeys([kw for s in curr_cluster_segs for kw in s.get("_matched_kws", [])]))
            clusters.append({
                "start": c_start,
                "end": c_end,
                "duration_sec": round(c_end - c_start, 1),
                "segment_count": len(curr_cluster_segs),
                "matched_keywords": kws,
                "segments": curr_cluster_segs,
            })
            curr_cluster_segs = [seg]

    if curr_cluster_segs:
        c_start = float(curr_cluster_segs[0].get("start", 0.0))
        c_end = float(curr_cluster_segs[-1].get("end", 0.0))
        kws = list(dict.fromkeys([kw for s in curr_cluster_segs for kw in s.get("_matched_kws", [])]))
        clusters.append({
            "start": c_start,
            "end": c_end,
            "duration_sec": round(c_end - c_start, 1),
            "segment_count": len(curr_cluster_segs),
            "matched_keywords": kws,
            "segments": curr_cluster_segs,
        })

    return clusters


def calculate_merged_coverage(
    intervals: List[Tuple[float, float]],
    total_duration_sec: float,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """计算多个区间的合并覆盖时长与覆盖率"""
    if not intervals or total_duration_sec <= 0:
        return 0.0, 0.0, []

    sorted_inv = sorted(intervals, key=lambda x: x[0])
    merged: List[List[float]] = []

    for a, b in sorted_inv:
        a_clamped = max(0.0, a)
        b_clamped = min(total_duration_sec, b)
        if a_clamped >= b_clamped:
            continue
        if merged and a_clamped <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b_clamped)
        else:
            merged.append([a_clamped, b_clamped])

    covered_sec = sum(b - a for a, b, in merged)
    coverage_ratio = covered_sec / total_duration_sec
    return covered_sec, coverage_ratio, [(a, b) for a, b in merged]


def generate_question_index_md(
    total_duration_sec: float,
    clusters: List[Dict[str, Any]],
    question_events: List[Dict[str, Any]],
    must_intervals: List[Dict[str, Any]],
    coverage_sec: float,
    coverage_ratio: float,
    merged_intervals: List[Tuple[float, float]],
) -> str:
    """生成符合规范要求的 question-index.md 文本"""
    lines: List[str] = [
        "# 面试录屏问题索引与精转候选区",
        "",
        f"- **视频总时长**: {format_hms(total_duration_sec)} ({total_duration_sec:.1f}s)",
        f"- **候选精转区间数**: {len(merged_intervals)} 个",
        f"- **合并总覆盖**: {format_hms(coverage_sec)} ({coverage_sec:.1f}s) / 占比 **{coverage_ratio:.1%}**",
        "",
    ]

    if coverage_ratio > 0.4:
        lines.extend([
            "> [!WARNING]",
            f"> 候选区间总覆盖率 ({coverage_ratio:.1%}) 已超过 40% 自动精转阈值！",
            "> 按规范要求停止自动全量 Turbo，提供以下三种选项供用户决策：",
            "> - **选项 a**: 全程精转（需用户明确批准）",
            "> - **选项 b**: 只精转高密度主题簇（命中段数 Top 簇）",
            "> - **选项 c**: 用户根据本索引手选指定区间",
            "",
        ])

    lines.append("## 一、主题簇视图 (Topic Clusters)")
    lines.append("")
    for i, c in enumerate(clusters, 1):
        lines.append(f"### 簇 {i}: [{format_hms(c['start'])} - {format_hms(c['end'])}] (时长: {c['duration_sec']}s, 命中 {c['segment_count']} 段)")
        lines.append(f"- **命中关键词**: {', '.join(c['matched_keywords'])}")
        lines.append("- **核心片段摘录**:")
        for s in c["segments"][:5]:  # 前 5 段
            lines.append(f"  - `[{format_hms(s['start'])}]` {s.get('text', '').strip()}")
        if len(c["segments"]) > 5:
            lines.append(f"  - *(其余 {len(c['segments']) - 5} 段略)*")
        lines.append("")

    lines.append("## 二、问句事件视图 (Question Events)")
    lines.append("")
    for i, qe in enumerate(question_events, 1):
        lines.append(f"{i}. `[{format_hms(qe['start'])}]` **{qe['reason']}**: {qe['text']}")
    lines.append("")

    lines.append("## 三、必须包含区间 (Must-Include)")
    lines.append("")
    for mi in must_intervals:
        lines.append(f"- **{mi['name']}**: [{format_hms(mi['start'])} - {format_hms(mi['end'])}] ({mi['reason']})")
    lines.append("")

    lines.append("## 四、最终合并建议精转区间")
    lines.append("")
    for i, (a, b) in enumerate(merged_intervals, 1):
        lines.append(f"- **区间 {i}**: `{format_hms(a)} - {format_hms(b)}` (`{a:.2f},{b:.2f}`，时长 {b-a:.1f}s)")
    lines.append("")

    return "\n".join(lines)


def build_question_index(
    transcript_json_path: str | Path,
    output_dir: str | Path,
    config: AnalyzerConfig,
    total_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Stage 4 构建问题索引完整算法流程"""
    t_path = Path(transcript_json_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(t_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])

    if not segments:
        raise ValueError("转写 JSON 中没有任何有效 segments！")

    if total_duration_sec is None:
        total_duration_sec = float(segments[-1].get("end", 0.0))

    ai_kws = config.indexing.ai_coding_keywords
    proj_kws = config.indexing.project_keywords
    q_pats = config.indexing.question_patterns
    rev_pats = config.indexing.reverse_question_patterns
    silence_thresh = config.thresholds.silence_rms_threshold

    # 1. 扫描段落关键词与问句
    hit_segments: List[Dict[str, Any]] = []
    question_events: List[Dict[str, Any]] = []

    for seg in segments:
        text = seg.get("text", "")
        rms = seg.get("_rms")

        # 关键词匹配
        kws = match_keywords(text, ai_kws, proj_kws)
        if kws:
            seg_copy = dict(seg)
            seg_copy["_matched_kws"] = kws
            hit_segments.append(seg_copy)

        # 扩展问句匹配
        is_q, q_reason = is_question_candidate(text, q_pats, rms=rms, silence_threshold=silence_thresh)
        if is_q:
            question_events.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text.strip(),
                "reason": q_reason,
            })

    # 2. 主题簇密度聚类 (< 180s 间隔)
    clusters = cluster_segments_by_density(hit_segments, max_gap_sec=180.0)

    # 3. 必须包含区间
    must_intervals = find_must_include_intervals(
        segments,
        total_duration_sec,
        reverse_patterns=rev_pats,
        silence_threshold=silence_thresh,
    )

    # 4. 汇总所有待精转区间
    candidate_intervals: List[Tuple[float, float]] = []
    for c in clusters:
        candidate_intervals.append((c["start"], c["end"]))
    for mi in must_intervals:
        candidate_intervals.append((mi["start"], mi["end"]))

    covered_sec, coverage_ratio, merged_intervals = calculate_merged_coverage(
        candidate_intervals,
        total_duration_sec=total_duration_sec,
    )

    # 5. 生成 question-index.md
    md_content = generate_question_index_md(
        total_duration_sec=total_duration_sec,
        clusters=clusters,
        question_events=question_events,
        must_intervals=must_intervals,
        coverage_sec=covered_sec,
        coverage_ratio=coverage_ratio,
        merged_intervals=merged_intervals,
    )

    index_md_path = out_dir / "question-index.md"
    with open(index_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "question_index_path": str(index_md_path),
        "total_duration_sec": total_duration_sec,
        "clusters": clusters,
        "question_events": question_events,
        "must_intervals": must_intervals,
        "merged_intervals": merged_intervals,
        "coverage_seconds": covered_sec,
        "coverage_ratio": coverage_ratio,
        "needs_user_menu": coverage_ratio > config.thresholds.coverage_auto,
    }
