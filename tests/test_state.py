"""状态管理与断点恢复单元测试"""

from pathlib import Path

from analyze_interview.state import ExecutionStage, ExecutionState, get_default_output_dir


def test_state_lifecycle(tmp_path: Path):
    video_file = tmp_path / "interview_demo.mp4"
    video_file.write_text("dummy video")
    out_dir = tmp_path / "interview_demo_analysis"

    state = ExecutionState(output_dir=out_dir, video_path=video_file)
    assert state.current_stage == ExecutionStage.STAGE_0_PRECHECK
    assert not state.is_stage_completed(ExecutionStage.STAGE_0_PRECHECK)

    # 标记完成
    state.mark_stage_complete(ExecutionStage.STAGE_0_PRECHECK, next_stage=ExecutionStage.STAGE_1_ENV_INSTALL)
    assert state.is_stage_completed(ExecutionStage.STAGE_0_PRECHECK)
    assert state.current_stage == ExecutionStage.STAGE_1_ENV_INSTALL

    # 登记产物
    art_file = out_dir / "source-info.json"
    art_file.write_text("{}")
    state.record_artifact("source_info", art_file)
    assert "source_info" in state.artifacts

    # 记录错误
    state.record_error(ExecutionStage.STAGE_1_ENV_INSTALL, "Missing dependencies")
    assert state.last_error is not None
    assert "Missing dependencies" in state.last_error

    # 重新加载状态文件核验持久化
    state_reloaded = ExecutionState(output_dir=out_dir, video_path=video_file)
    assert state_reloaded.is_stage_completed(ExecutionStage.STAGE_0_PRECHECK)
    assert "source_info" in state_reloaded.artifacts
    assert state_reloaded.last_error == state.last_error

    # 核验日志文件
    log_content = (out_dir / "execution-log.txt").read_text(encoding="utf-8")
    assert "阶段完成: stage_0_precheck" in log_content
    assert "Missing dependencies" in log_content


def test_get_default_output_dir():
    v = Path("/tmp/videos/2026-08-30.mp4")
    out = get_default_output_dir(v)
    assert out.name == "2026-08-30_analysis"
    assert out.parent == v.resolve().parent
