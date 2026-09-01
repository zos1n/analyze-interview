"""
Stage 3.5 转写病理检测与定向修复/确定性片段合并模块。
提供裸 PCM RMS 静音幻觉检测、重复循环检测、断档检测、中文幻觉过滤与确定性合并算法。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from analyze_interview.audio import load_pcm_f32le_16k
from analyze_interview.config import AnalyzerConfig

# 强特征（命中即无条件判定为幻觉排除）
STRONG_HALLUCINATIONS = re.compile(
    r"(请不吝点赞|明镜与点点|谢谢观看|欢迎订阅|打赏支持|投币|关注本频道|下期再见)",
    re.IGNORECASE,
)

# 弱特征（仅当短句且无技术上下文时判定为幻觉）
WEAK_HALLUCINATIONS = re.compile(r"(字幕|点赞|关注|订阅|转发)", re.IGNORECASE)

# 技术上下文白名单
TECH_CONTEXT_WORDS = ("组件", "接口", "功能", "模块", "字符串", "业务", "代码", "设计", "逻辑", "状态")

# 兼容别名
CHINESE_HALLUCINATION_PATTERN = STRONG_HALLUCINATIONS


def compute_segment_rms(
    pcm_data: np.ndarray,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
) -> float:
    """计算指定时间区间的 PCM 均方根 (RMS) 能量"""
    if len(pcm_data) == 0:
        return 0.0
    start_idx = max(0, int(start_sec * sample_rate))
    end_idx = min(len(pcm_data), int(end_sec * sample_rate))
    if start_idx >= end_idx:
        return 0.0
    slice_data = pcm_data[start_idx:end_idx]
    return float(np.sqrt(np.mean(slice_data ** 2)))


def detect_silence_hallucinations(
    segments: List[Dict[str, Any]],
    pcm_data: np.ndarray,
    rms_threshold: float = 0.005,
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    """检测落在静音区（RMS < 阈值）的文本幻觉"""
    results: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", 0.0))
        rms = compute_segment_rms(pcm_data, start_sec, end_sec, sample_rate=sample_rate)
        seg["_rms"] = rms
        if rms < rms_threshold and len(seg.get("text", "").strip()) > 0:
            results.append({
                "segment_index": idx,
                "start": start_sec,
                "end": end_sec,
                "text": seg.get("text", "").strip(),
                "rms": round(rms, 6),
                "threshold": rms_threshold,
            })
    return results


def detect_repetition_loops(
    segments: List[Dict[str, Any]],
    pcm_data: np.ndarray,
    min_repeats: int = 3,
    min_chars: int = 8,
    rms_threshold: float = 0.005,
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    """检测 Whisper 常见重复循环病理（叠加 RMS 验证区分真口吃与静音幻觉）"""
    loops: List[Dict[str, Any]] = []
    if len(segments) < min_repeats:
        return loops

    i = 0
    while i < len(segments):
        curr_text = segments[i].get("text", "").strip()
        if len(curr_text) < min_chars:
            i += 1
            continue

        repeat_count = 1
        j = i + 1
        while j < len(segments):
            next_text = segments[j].get("text", "").strip()
            if next_text == curr_text:
                repeat_count += 1
                j += 1
            else:
                break

        if repeat_count >= min_repeats:
            start_sec = float(segments[i].get("start", 0.0))
            end_sec = float(segments[j - 1].get("end", 0.0))
            rms = compute_segment_rms(pcm_data, start_sec, end_sec, sample_rate=sample_rate)
            is_silence = rms < rms_threshold

            loops.append({
                "start_segment_index": i,
                "end_segment_index": j - 1,
                "start": start_sec,
                "end": end_sec,
                "repeat_count": repeat_count,
                "text": curr_text,
                "rms": round(rms, 6),
                "classification": "silence_hallucination" if is_silence else "genuine_repetition",
            })
            i = j
        else:
            i += 1

    return loops


def detect_timestamp_gaps(
    segments: List[Dict[str, Any]],
    pcm_data: np.ndarray,
    gap_threshold: float = 30.0,
    rms_threshold: float = 0.005,
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    """检测时间戳断档（相邻段间隔 > 阈值且该处非静音，疑似漏转）"""
    gaps: List[Dict[str, Any]] = []
    for i in range(len(segments) - 1):
        curr_end = float(segments[i].get("end", 0.0))
        next_start = float(segments[i + 1].get("start", 0.0))
        gap_sec = next_start - curr_end
        if gap_sec > gap_threshold:
            rms = compute_segment_rms(pcm_data, curr_end, next_start, sample_rate=sample_rate)
            if rms >= rms_threshold:
                gaps.append({
                    "prev_segment_index": i,
                    "next_segment_index": i + 1,
                    "gap_start": curr_end,
                    "gap_end": next_start,
                    "gap_duration_sec": round(gap_sec, 2),
                    "rms": round(rms, 6),
                })
    return gaps


def detect_chinese_hallucination_phrases(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """检测常见中文幻觉短语（采用强特征/弱特征+技术上下文白名单分级判定，防误杀）"""
    hits: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        # 1. 强特征命中：无条件判定为幻觉
        strong_match = STRONG_HALLUCINATIONS.search(text)
        if strong_match:
            hits.append({
                "segment_index": idx,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
                "matched_keyword": strong_match.group(0),
                "level": "strong",
            })
            continue

        # 2. 弱特征命中：长度 < 25 且无技术上下文时判定为幻觉
        weak_match = WEAK_HALLUCINATIONS.search(text)
        if weak_match:
            has_tech_context = any(w in text for w in TECH_CONTEXT_WORDS)
            if len(text) < 25 and not has_tech_context:
                hits.append({
                    "segment_index": idx,
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": text,
                    "matched_keyword": weak_match.group(0),
                    "level": "weak",
                })

    return hits


def run_pathology_check(
    transcript_json_path: str | Path,
    asr_flac_path: str | Path,
    config: AnalyzerConfig,
    ffmpeg_path: Optional[str] = None,
) -> Dict[str, Any]:
    """对转写产物执行完整的病理检测流程"""
    t_path = Path(transcript_json_path).expanduser().resolve()
    flac_path = Path(asr_flac_path).expanduser().resolve()

    with open(t_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])

    ff_bin = ffmpeg_path or config.runtime.ffmpeg
    pcm_data = load_pcm_f32le_16k(flac_path, channels=1, ffmpeg_path=ff_bin)

    silence_hits = detect_silence_hallucinations(
        segments,
        pcm_data,
        rms_threshold=config.thresholds.silence_rms_threshold,
    )
    loop_hits = detect_repetition_loops(
        segments,
        pcm_data,
        min_repeats=config.thresholds.loop_repeat_min,
        min_chars=config.thresholds.loop_text_min_chars,
        rms_threshold=config.thresholds.silence_rms_threshold,
    )
    gap_hits = detect_timestamp_gaps(
        segments,
        pcm_data,
        gap_threshold=config.thresholds.timestamp_gap_seconds,
        rms_threshold=config.thresholds.silence_rms_threshold,
    )
    phrase_hits = detect_chinese_hallucination_phrases(segments)

    has_bad_loops = any(h["classification"] == "silence_hallucination" or h["repeat_count"] >= 5 for h in loop_hits)

    return {
        "transcript_path": str(t_path),
        "total_segments": len(segments),
        "silence_hallucinations": silence_hits,
        "repetition_loops": loop_hits,
        "timestamp_gaps": gap_hits,
        "chinese_hallucinations": phrase_hits,
        "needs_repair": has_bad_loops,
    }


def deterministic_merge_segments(
    original_segments: List[Dict[str, Any]],
    repair_segments: List[Dict[str, Any]],
    repair_start_sec: float,
    repair_end_sec: float,
    window_padding_sec: float = 3.0,
) -> List[Dict[str, Any]]:
    """确定性片段合并算法（遵循规范阶段 3.5 与阶段 5 规则）：
    1. 修复窗口 = [repair_start - 3s, repair_end + 3s]。
    2. 落在修复窗口内的原段全部丢弃，由修复段替换。
    3. 修复段整段保留，不截断、不因跨界丢弃。
    4. 修复段优先：修复段整段跨度内覆盖到的窗口外原段同样丢弃，不产生双写。
    5. 窗口外且未被修复段覆盖的原段原样保留。
    6. 合并结果按时间戳升序重排。
    """
    win_start = max(0.0, repair_start_sec - window_padding_sec)
    win_end = repair_end_sec + window_padding_sec

    # 确定所有有效修复段的实际覆盖范围
    valid_repair_segs = [s for s in repair_segments if float(s.get("start", 0.0)) >= win_start - 0.5]
    if not valid_repair_segs:
        valid_repair_segs = repair_segments

    # 计算修复段整体跨度
    if valid_repair_segs:
        rep_span_start = min(float(s.get("start", 0.0)) for s in valid_repair_segs)
        rep_span_end = max(float(s.get("end", 0.0)) for s in valid_repair_segs)
    else:
        rep_span_start = win_start
        rep_span_end = win_end

    kept_original: List[Dict[str, Any]] = []
    for s in original_segments:
        s_start = float(s.get("start", 0.0))
        s_end = float(s.get("end", 0.0))

        # 规则2：落在修复窗口内的原段丢弃
        if (s_start >= win_start and s_start <= win_end) or (s_end >= win_start and s_end <= win_end):
            continue

        # 规则4：修复段优先，覆盖到的原段丢弃
        if not (s_end <= rep_span_start or s_start >= rep_span_end):
            continue

        kept_original.append(s)

    # 规则3：修复段整段保留
    merged = kept_original + valid_repair_segs

    # 规则6：按时间戳升序排序
    merged.sort(key=lambda x: float(x.get("start", 0.0)))

    # 重建 ID
    for idx, s in enumerate(merged):
        s["id"] = idx

    return merged
