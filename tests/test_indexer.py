"""问题索引与密度聚类算法单元测试"""

from analyze_interview.indexer import (
    calculate_merged_coverage,
    cluster_segments_by_density,
    format_hms,
    is_question_candidate,
    match_keywords,
)


def test_format_hms():
    assert format_hms(0) == "00:00"
    assert format_hms(65) == "01:05"
    assert format_hms(3665) == "01:01:05"


def test_match_keywords_ascii_word_boundary():
    ai_kws = ["AI", "Agent", "Coding"]
    proj_kws = ["项目", "优化"]

    # "AI" 不应命中 "said", "wait", "email" 等包含 ai 子串的单词
    assert match_keywords("He said we should wait for AI Agent", ai_kws, proj_kws) == ["AI", "Agent"]
    assert match_keywords("We need to wait for email response", ai_kws, proj_kws) == []

    # 中文关键词子串匹配
    assert match_keywords("负责核心业务前端项目性能优化", ai_kws, proj_kws) == ["项目", "优化"]


def test_is_question_candidate():
    q_patterns = ["为什么", "怎么", "如何", "有没有", "介绍一下", "具体说说"]

    # 句末疑问助词
    is_q, reason = is_question_candidate("有遇到过长列表卡顿吗？", q_patterns)
    assert is_q is True

    # 疑问代词
    is_q, reason = is_question_candidate("你们团队的开发排期大概多久？", q_patterns)
    assert is_q is True

    # 追问引导词
    is_q, reason = is_question_candidate("那你们在异常熔断时怎么处理的？", q_patterns)
    assert is_q is True

    # 排除短回应词
    is_q, _ = is_question_candidate("嗯", q_patterns)
    assert is_q is False

    is_q, _ = is_question_candidate("好的没问题", q_patterns)
    assert is_q is False

    # 排除解释性句子
    is_q, _ = is_question_candidate("其实我们主要采用了双缓冲队列机制", q_patterns)
    assert is_q is False


def test_cluster_segments_by_density():
    segs = [
        {"id": 0, "start": 10.0, "end": 20.0, "_matched_kws": ["AI"]},
        {"id": 1, "start": 50.0, "end": 60.0, "_matched_kws": ["Agent"]},     # 间隔 30s -> 同一簇
        {"id": 2, "start": 300.0, "end": 310.0, "_matched_kws": ["项目"]},    # 间隔 240s > 180s -> 新簇
    ]
    clusters = cluster_segments_by_density(segs, max_gap_sec=180.0)
    assert len(clusters) == 2
    assert clusters[0]["start"] == 10.0
    assert clusters[0]["end"] == 60.0
    assert clusters[0]["segment_count"] == 2
    assert clusters[1]["start"] == 300.0
    assert clusters[1]["end"] == 310.0


def test_calculate_merged_coverage():
    intervals = [(10.0, 50.0), (40.0, 80.0), (100.0, 150.0)]
    total_sec = 200.0

    covered_sec, ratio, merged = calculate_merged_coverage(intervals, total_sec)
    # (10-80s) = 70s, (100-150s) = 50s => 总覆盖 120s
    assert covered_sec == 120.0
    assert ratio == 0.60
    assert len(merged) == 2
