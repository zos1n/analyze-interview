"""关键问答稿生成与说话人归属单元测试"""

from analyze_interview.config import AnalyzerConfig
from analyze_interview.key_transcript import (
    build_key_transcript_md,
    classify_speech_tone,
    extract_key_qa_pipeline,
    generate_asr_term_table,
)


def test_classify_speech_tone():
    # 第一优先级：解释性/回答前缀 (长度 > 15 字强制为 answer)
    assert classify_speech_tone("其实我们当时在架构设计中主要通过双缓冲队列来解决消费堆积") == "answer"
    assert classify_speech_tone("因为我们这边主要负责的是大模型工作流编排系统的性能调优") == "answer"
    assert classify_speech_tone("具体来说，主要是在底层的 Web Worker 中处理高频数据计算") == "answer"

    # 第二优先级：典型追问句式
    assert classify_speech_tone("怎么去保证消费顺序？") == "question"
    assert classify_speech_tone("为什么会有这种强依赖呢？") == "question"
    assert classify_speech_tone("那你们具体是如何做容灾降级的？") == "question"
    assert classify_speech_tone("是不是应该先校验 token 状态？") == "question"

    # 第三优先级：默认陈述句
    assert classify_speech_tone("我们在底层做了消息 ACK 确认机制。") == "answer"
    assert classify_speech_tone("大概就是这个方案。") == "answer"


def test_generate_asr_term_table():
    term_map = {"reg": "RAG", "Cloud Code": "Claude Code"}
    table_md = generate_asr_term_table(term_map)
    assert "| `reg` | **RAG** |" in table_md
    assert "| `Cloud Code` | **Claude Code** |" in table_md


def test_extract_key_qa_pipeline(sample_config: AnalyzerConfig, sample_segments: list):
    q_events = [
        {"start": 3.0, "end": 8.0, "text": "你能先简单介绍一下你自己吗？", "reason": "模式:介绍一下"},
        {"start": 26.0, "end": 35.0, "text": "那你们在项目中SSE流式通信和打字机渲染是怎么做性能优化的？", "reason": "模式:怎么"},
    ]

    qa_data = extract_key_qa_pipeline(
        turbo_or_full_segments=sample_segments,
        question_events=q_events,
        total_duration_sec=120.0,
        config=sample_config,
    )

    assert len(qa_data) == 2
    # 第一题：自我介绍
    assert qa_data[0]["interviewer_lines"][0]["text"] == "你能先简单介绍一下你自己吗？"
    assert len(qa_data[0]["candidate_lines"]) >= 1
    assert "张三" in qa_data[0]["candidate_lines"][0]["text"]

    # 第二题：性能优化
    assert "SSE流式通信" in qa_data[1]["interviewer_lines"][0]["text"]
    assert len(qa_data[1]["candidate_lines"]) >= 1
    assert "调度缓冲区" in qa_data[1]["candidate_lines"][0]["text"]


def test_extract_key_qa_pipeline_split_tracks(sample_config: AnalyzerConfig):
    q_events = [
        {"start": 10.0, "end": 20.0, "text": "介绍一下你们的微前端架构方案？", "reason": "模式:介绍一下"},
    ]
    interviewer_segs = [
        {"start": 10.0, "end": 20.0, "text": "介绍一下你们的微前端架构方案？"},
        {"start": 50.0, "end": 55.0, "text": "那沙箱隔离是怎么处理的？"},  # 追问
    ]
    candidate_segs = [
        {"start": 21.0, "end": 45.0, "text": "我们基于 qiankun 做了改造，并封装了应用通信协议。"},
        {"start": 56.0, "end": 70.0, "text": "通过 Proxy 对全局 window 对象做了代理隔离。"},
    ]

    qa_data = extract_key_qa_pipeline(
        turbo_or_full_segments=interviewer_segs,
        question_events=q_events,
        total_duration_sec=100.0,
        config=sample_config,
        is_split_tracks=True,
        candidate_track_segments=candidate_segs,
    )

    assert len(qa_data) == 1
    assert qa_data[0]["interviewer_lines"][0]["text"] == "介绍一下你们的微前端架构方案？"
    assert len(qa_data[0]["candidate_lines"]) == 2
    assert "qiankun" in qa_data[0]["candidate_lines"][0]["text"]
    assert "Proxy" in qa_data[0]["candidate_lines"][1]["text"]
    assert len(qa_data[0]["followup_lines"]) == 1
    assert "沙箱隔离" in qa_data[0]["followup_lines"][0]["text"]


def test_build_key_transcript_md():
    questions_data = [
        {
            "topic": "自我介绍",
            "start": 0.0,
            "end": 30.0,
            "interviewer_lines": [{"time": 0.0, "text": "请介绍一下自己"}],
            "candidate_lines": [{"time": 5.0, "text": "我是张三，5年前端经验"}],
            "followup_lines": [],
            "signals": ["交互顺畅"],
        }
    ]
    term_map = {"reg": "RAG"}
    md = build_key_transcript_md(questions_data, term_map)
    assert "# 面试逐题关键问答稿 (Key Transcript)" in md
    assert "## 问题 1：自我介绍" in md
    assert "我是张三，5年前端经验" in md
