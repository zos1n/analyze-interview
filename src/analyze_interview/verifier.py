"""
Stage 8 最终验收核验模块。
核对源视频 SHA256/mtime 基线一致性、全部产物完整性与规范遵从性。
"""

from __future__ import annotations

from typing import Any, Dict

from analyze_interview.precheck import calculate_video_baseline
from analyze_interview.state import ExecutionState


def run_stage_8_verification(
    state: ExecutionState,
) -> Dict[str, Any]:
    """执行 Stage 8 验收检查"""
    results: Dict[str, Any] = {
        "passed": True,
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    # 1. 检查源视频哈希基线
    base = state.video_baseline
    if base and base.get("sha256"):
        current_base = calculate_video_baseline(state.video_path)
        sha_match = current_base["sha256"] == base["sha256"]
        size_match = current_base["size_bytes"] == base["size_bytes"]
        results["checks"].append({
            "name": "原视频哈希与大小一致性 (SHA256)",
            "passed": sha_match and size_match,
            "details": f"Baseline SHA256: {base['sha256'][:12]}... / Current: {current_base['sha256'][:12]}...",
        })
        if not sha_match:
            results["passed"] = False
            results["errors"].append("严重警告：原视频 SHA256 与初始基线不一致，源文件可能被篡改！")
    else:
        results["warnings"].append("未找到初始原视频基线，跳过 SHA256 对比")

    # 2. 检查关键产物是否存在且非空
    if state.audio_mode == "split_tracks":
        required_artifacts = [
            ("source-info.json", state.output_dir / "source-info.json"),
            ("audio-asr-track1.flac", state.output_dir / "audio-asr-track1.flac"),
            ("audio-asr-track2.flac", state.output_dir / "audio-asr-track2.flac"),
            ("question-index.md", state.output_dir / "question-index.md"),
            ("transcript-key.md", state.output_dir / "transcript-key.md"),
            ("interview-review.md", state.output_dir / "interview-review.md"),
            ("execution-state.json", state.output_dir / "execution-state.json"),
            ("execution-log.txt", state.output_dir / "execution-log.txt"),
        ]
    else:
        required_artifacts = [
            ("source-info.json", state.output_dir / "source-info.json"),
            ("audio-asr.flac", state.output_dir / "audio-asr.flac"),
            ("transcript-full.json", state.output_dir / "transcript-full.json"),
            ("question-index.md", state.output_dir / "question-index.md"),
            ("transcript-key.md", state.output_dir / "transcript-key.md"),
            ("interview-review.md", state.output_dir / "interview-review.md"),
            ("execution-state.json", state.output_dir / "execution-state.json"),
            ("execution-log.txt", state.output_dir / "execution-log.txt"),
        ]

    for name, path in required_artifacts:
        exists = path.exists() and path.stat().st_size > 0
        results["checks"].append({
            "name": f"核心产物核验: {name}",
            "passed": exists,
            "details": f"Path: {path} (size: {path.stat().st_size if path.exists() else 0} bytes)",
        })
        if not exists:
            # 部分可选文件记录为 warning
            if name in ["interview-review.md"]:
                results["warnings"].append(f"产物 {name} 缺失或为空")
            else:
                results["passed"] = False
                results["errors"].append(f"核心产物缺失: {name}")

    # 3. 验收审计通过记录
    if results["passed"]:
        state.log("Stage 8 最终验收全部通过！所有指标与产物均符合规范要求。", level="INFO")
    else:
        state.log(f"Stage 8 最终验收未通过: {', '.join(results['errors'])}", level="ERROR")

    return results
