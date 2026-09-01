"""流水线覆盖率补录算法、Stage 8 验收核验与 7 维度复盘骨架单元测试"""

from pathlib import Path

from analyze_interview.pipeline import compensate_turbo_coverage
from analyze_interview.reviewer import generate_review_skeleton
from analyze_interview.state import ExecutionState
from analyze_interview.verifier import run_stage_8_verification


def test_compensate_turbo_coverage():
    # 模拟 3 个问句事件
    q_events = [
        {"start": 100.0, "end": 110.0, "text": "问题1：为什么做微前端？"},   # 覆盖
        {"start": 300.0, "end": 310.0, "text": "问题2：怎么优化打包体积？"}, # 漂移 15s -> 仍然覆盖 (<= 20s)
        {"start": 600.0, "end": 610.0, "text": "问题3：SSE 流式如何重连？"}, # 未覆盖
        {"start": 680.0, "end": 690.0, "text": "问题4：重连后的消息去重？"}, # 未覆盖，与问题3间隔 70s <= 120s -> 聚为一簇
    ]

    turbo_segments = [
        {"start": 95.0, "end": 120.0, "text": "问题1的精转内容"},
        {"start": 325.0, "end": 340.0, "text": "问题2在 Turbo 漂移后的精转内容"},  # 325 <= 310 + 20 -> 覆盖
        {"start": 500.0, "end": 510.0, "text": "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"}, # 幻觉段，被过滤不参与覆盖计数
    ]

    current_intervals = [(90.0, 130.0), (320.0, 350.0)]
    total_duration_sec = 1000.0

    comp_intervals, total_ratio, uncovered = compensate_turbo_coverage(
        turbo_segments=turbo_segments,
        question_events=q_events,
        total_duration_sec=total_duration_sec,
        current_intervals=current_intervals,
        coverage_threshold=0.40,
    )

    # 未覆盖的应为问题3和问题4
    assert len(uncovered) == 2
    assert uncovered[0]["text"] == "问题3：SSE 流式如何重连？"
    assert uncovered[1]["text"] == "问题4：重连后的消息去重？"

    # 问题3和问题4间隔 70s <= 120s，应聚类为一个补选区间：
    # cluster start = 600.0 - 10.0 = 590.0
    # cluster end = 690.0 + 60.0 = 750.0
    assert len(comp_intervals) == 1
    assert comp_intervals[0] == (590.0, 750.0)

    # 总覆盖时长：(130-90) + (350-320) + (750-590) = 40 + 30 + 160 = 230s => 23.0%
    assert 0.229 < total_ratio < 0.231


def test_reviewer_7_dimensions_skeleton(tmp_path: Path):
    questions_data = [
        {"topic": "自适应流式调度", "start": 10.0, "end": 60.0},
    ]
    out_review_path = tmp_path / "interview-review.md"
    generate_review_skeleton(questions_data, out_review_path)

    assert out_review_path.exists()
    content = out_review_path.read_text(encoding="utf-8")

    # 验证 7 维度完全对齐
    assert "### 1. 【考察目标】" in content
    assert "### 2. 【回答覆盖内容】" in content
    assert "### 3. 【缺失内容与风险】" in content
    assert "### 4. 【客观现场信号】" in content
    assert "### 5. 【失分判定】" in content
    assert "明确失分 / 隐性失分 / 无明显问题" in content
    assert "### 6. 【更好的回答结构】" in content
    assert "### 7. 【对结果影响与置信度】" in content


def test_stage_8_verifier_dynamic_artifacts(tmp_path: Path):
    video_file = tmp_path / "mock_video.mp4"
    video_file.write_bytes(b"mock_video_bytes")

    out_dir = tmp_path / "mock_video_analysis"
    out_dir.mkdir()

    state = ExecutionState(output_dir=out_dir, video_path=video_file)

    # 构造通用文件
    (out_dir / "source-info.json").write_text("{}", encoding="utf-8")
    (out_dir / "question-index.md").write_text("# Index", encoding="utf-8")
    (out_dir / "transcript-key.md").write_text("# Key", encoding="utf-8")
    (out_dir / "interview-review.md").write_text("# Review", encoding="utf-8")
    (out_dir / "execution-state.json").write_text("{}", encoding="utf-8")
    (out_dir / "execution-log.txt").write_text("Log", encoding="utf-8")

    # 1. 单轨模式：需要 audio-asr.flac 和 transcript-full.json
    state.audio_mode = "single_track"
    (out_dir / "audio-asr.flac").write_bytes(b"fake_flac")
    (out_dir / "transcript-full.json").write_text("{}", encoding="utf-8")

    # 跳过 sha 对比测试产物核验
    state.video_baseline = None
    res_single = run_stage_8_verification(state)
    assert res_single["passed"] is True

    # 2. 分轨模式：需要 audio-asr-track1.flac 和 audio-asr-track2.flac
    state.audio_mode = "split_tracks"
    res_split_fail = run_stage_8_verification(state)
    assert res_split_fail["passed"] is False  # 缺失 track1/track2

    (out_dir / "audio-asr-track1.flac").write_bytes(b"fake_flac_1")
    (out_dir / "audio-asr-track2.flac").write_bytes(b"fake_flac_2")
    res_split_pass = run_stage_8_verification(state)
    assert res_split_pass["passed"] is True
