"""转写病理检测与确定性片段合并算法单元测试"""

import numpy as np

from analyze_interview.pathology import (
    compute_segment_rms,
    detect_chinese_hallucination_phrases,
    detect_repetition_loops,
    detect_silence_hallucinations,
    deterministic_merge_segments,
)


def test_compute_segment_rms(synthetic_pcm_16k: np.ndarray):
    # 0.0s - 2.0s 为静音
    rms_silence = compute_segment_rms(synthetic_pcm_16k, 0.0, 2.0)
    assert rms_silence == 0.0

    # 4.0s - 6.0s 为振幅 0.2 的正弦波，理论 RMS 约为 0.2 / sqrt(2) ≈ 0.1414
    rms_speech = compute_segment_rms(synthetic_pcm_16k, 4.0, 6.0)
    assert 0.13 < rms_speech < 0.15


def test_detect_silence_hallucinations(synthetic_pcm_16k: np.ndarray):
    segs = [
        {"id": 0, "start": 0.5, "end": 2.0, "text": "落在静音区的幻觉文字"},
        {"id": 1, "start": 3.5, "end": 5.5, "text": "正常真实语音"},
    ]
    hits = detect_silence_hallucinations(segs, synthetic_pcm_16k, rms_threshold=0.005)
    assert len(hits) == 1
    assert hits[0]["segment_index"] == 0
    assert hits[0]["text"] == "落在静音区的幻觉文字"


def test_detect_repetition_loops(synthetic_pcm_16k: np.ndarray):
    # 构造 4 次连续重复文本（长度 >= 8）
    segs = [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "我看你不喜歡我"},
        {"id": 1, "start": 1.0, "end": 2.0, "text": "我看你不喜歡我"},
        {"id": 2, "start": 2.0, "end": 3.0, "text": "我看你不喜歡我"},
        {"id": 3, "start": 3.0, "end": 4.0, "text": "我看你不喜歡我"},
        {"id": 4, "start": 4.0, "end": 5.0, "text": "正常说话内容"},
    ]
    loops = detect_repetition_loops(
        segs, synthetic_pcm_16k, min_repeats=3, min_chars=6, rms_threshold=0.005
    )
    assert len(loops) == 1
    assert loops[0]["repeat_count"] == 4
    assert loops[0]["text"] == "我看你不喜歡我"


def test_detect_chinese_hallucination_phrases():
    segs = [
        # 强特征：无条件识别为幻觉
        {"id": 0, "start": 0.0, "end": 2.0, "text": "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"},
        {"id": 1, "start": 2.0, "end": 4.0, "text": "谢谢观看，欢迎订阅，下期再见"},
        # 弱特征 + 无技术上下文 + 短句 (<25字) -> 识别为幻觉
        {"id": 2, "start": 5.0, "end": 6.0, "text": "点赞关注"},
        # 弱特征 + 包含技术上下文白名单 -> 防误杀，保留为正常文本
        {"id": 3, "start": 7.0, "end": 9.0, "text": "我们负责实现点赞功能模块"},
        {"id": 4, "start": 10.0, "end": 12.0, "text": "ES6 模板字符串的底层解析逻辑"},
        {"id": 5, "start": 13.0, "end": 15.0, "text": "关注组件的状态设计与业务代码"},
        # 正常技术文本
        {"id": 6, "start": 16.0, "end": 18.0, "text": "我们在做大模型对话的流式渲染架构"},
    ]
    hits = detect_chinese_hallucination_phrases(segs)
    # 应只命中 segs 0, 1, 2
    assert len(hits) == 3
    hit_indices = [h["segment_index"] for h in hits]
    assert hit_indices == [0, 1, 2]
    assert hits[0]["level"] == "strong"
    assert hits[1]["level"] == "strong"
    assert hits[2]["level"] == "weak"


def test_deterministic_merge_segments():
    # 模拟原段落 0-100s
    orig_segs = [
        {"id": 0, "start": 0.0, "end": 10.0, "text": "开头段落"},
        {"id": 1, "start": 15.0, "end": 25.0, "text": "损坏段落1"},
        {"id": 2, "start": 25.0, "end": 35.0, "text": "损坏段落2"},
        {"id": 3, "start": 35.0, "end": 45.0, "text": "损坏段落3"},
        {"id": 4, "start": 50.0, "end": 60.0, "text": "正常后续段落"},
    ]

    # 修复段落 (修复 15s - 45s, 修复窗口 [12s, 48s])
    # 假设修复段落轻微越界到 49.5s，验证整段保留不截断
    repair_segs = [
        {"id": 0, "start": 14.5, "end": 28.0, "text": "修复后的完整句子1"},
        {"id": 1, "start": 28.5, "end": 49.5, "text": "修复后的完整句子2（越界保留）"},
    ]

    merged = deterministic_merge_segments(
        original_segments=orig_segs,
        repair_segments=repair_segs,
        repair_start_sec=15.0,
        repair_end_sec=45.0,
        window_padding_sec=3.0,
    )

    # 检查结果：
    # 1. 损坏段落 1, 2, 3 全部被剔除
    # 2. 开头段落 (0-10s) 与 正常后续段落 (50-60s) 保留
    # 3. 修复段落 2 个整段保留
    # 4. 时间戳升序排列且 id 重建
    texts = [s["text"] for s in merged]
    assert "开头段落" in texts
    assert "修复后的完整句子1" in texts
    assert "修复后的完整句子2（越界保留）" in texts
    assert "损坏段落1" not in texts
    assert "损坏段落2" not in texts
    assert "损坏段落3" not in texts
    assert len(merged) == 4
    assert merged[0]["start"] == 0.0
    assert merged[1]["start"] == 14.5
    assert merged[2]["start"] == 28.5
    assert merged[3]["start"] == 50.0
