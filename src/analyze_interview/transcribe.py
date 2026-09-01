"""
Stage 3 & Stage 5 转写模块：基于 Apple Silicon MLX 的 Whisper Small 全量粗转与 Whisper Turbo 局部精转。
严格遵循 mlx-whisper 0.4.3 参数规范、单 clip 隔离与整数边界命名。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from analyze_interview.config import AnalyzerConfig


def _get_mlx_whisper_bin(config: AnalyzerConfig) -> str:
    """获取 mlx_whisper 可执行文件路径"""
    venv_p = Path(config.runtime.venv_path).expanduser()
    bin_path = venv_p / "bin" / "mlx_whisper"
    if bin_path.exists() and os.access(bin_path, os.X_OK):
        return str(bin_path)
    return "mlx_whisper"


def run_small_transcribe(
    audio_path: str | Path,
    output_dir: str | Path,
    config: AnalyzerConfig,
    output_name: str = "transcript-full",
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """执行 Whisper Small 全量粗转 (Stage 3)"""
    audio_p = Path(audio_path).expanduser().resolve()
    out_d = Path(output_dir).expanduser().resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    mlx_bin = _get_mlx_whisper_bin(config)
    cmd = [
        mlx_bin,
        str(audio_p),
        "--model", config.models.small,
        "--language", "zh",
        "--task", "transcribe",
        "--output-format", "all",
        "--output-dir", str(out_d),
        "--output-name", output_name,
        "--initial-prompt", config.interview.vocab_prompt,
        "--verbose", "False",
    ]

    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        timeout=1800,  # 30 分钟超时
    )

    if res.returncode != 0:
        raise RuntimeError(
            f"Whisper Small 转写失败 (退出码 {res.returncode}):\n{res.stderr or res.stdout}"
        )

    expected_json = out_d / f"{output_name}.json"
    if not expected_json.exists():
        raise FileNotFoundError(f"转写产物 JSON 不存在: {expected_json}")

    return {
        "output_json": str(expected_json),
        "output_srt": str(out_d / f"{output_name}.srt"),
        "output_txt": str(out_d / f"{output_name}.txt"),
        "model": config.models.small,
    }


def run_turbo_clip(
    audio_path: str | Path,
    output_dir: str | Path,
    clip_index: int,
    start_sec: float,
    end_sec: float,
    config: AnalyzerConfig,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """执行 Whisper Large V3 Turbo 单区间精转 (Stage 5)
    严格遵循每个区间单独一条命令执行，避开 mlx-whisper 0.4.3 多区间连续转录 bug。
    """
    audio_p = Path(audio_path).expanduser().resolve()
    out_d = Path(output_dir).expanduser().resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    clip_name = f"transcript-turbo-clip-{clip_index}"
    clip_timestamps_arg = f"{start_sec:.2f},{end_sec:.2f}"

    mlx_bin = _get_mlx_whisper_bin(config)
    cmd = [
        mlx_bin,
        str(audio_p),
        "--model", config.models.turbo,
        "--language", "zh",
        "--task", "transcribe",
        "--output-format", "json",
        "--output-dir", str(out_d),
        "--output-name", clip_name,
        "--word-timestamps", "True",
        "--clip-timestamps", clip_timestamps_arg,
        "--initial-prompt", config.interview.vocab_prompt,
        "--condition-on-previous-text", "False",
        "--verbose", "False",
    ]

    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        timeout=600,
    )

    if res.returncode != 0:
        raise RuntimeError(
            f"Whisper Turbo 区间 [{clip_timestamps_arg}] 精转失败 (退出码 {res.returncode}):\n{res.stderr or res.stdout}"
        )

    expected_json = out_d / f"{clip_name}.json"
    if not expected_json.exists():
        raise FileNotFoundError(f"Turbo clip 转写产物不存在: {expected_json}")

    return {
        "clip_index": clip_index,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "output_json": str(expected_json),
    }


def run_repair_transcribe(
    audio_path: str | Path,
    output_dir: str | Path,
    model_type: str,  # "small" | "turbo"
    start_sec: float,
    end_sec: float,
    config: AnalyzerConfig,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """执行病理定向修复转写
    修复文件名强制使用整数边界，如 transcript-turbo-repair-88-159.json，
    防止 mlx-whisper 将末尾 .NN 识别为扩展名改写。
    """
    audio_p = Path(audio_path).expanduser().resolve()
    out_d = Path(output_dir).expanduser().resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    r_start = int(round(start_sec))
    r_end = int(round(end_sec))
    prefix = "small" if model_type == "small" else "turbo"
    output_name = f"transcript-{prefix}-repair-{r_start}-{r_end}"
    model_name = config.models.small if model_type == "small" else config.models.turbo

    # clip 仍传浮点精确值
    clip_timestamps_arg = f"{start_sec:.2f},{end_sec:.2f}"

    mlx_bin = _get_mlx_whisper_bin(config)
    cmd = [
        mlx_bin,
        str(audio_p),
        "--model", model_name,
        "--language", "zh",
        "--task", "transcribe",
        "--output-format", "json",
        "--output-dir", str(out_d),
        "--output-name", output_name,
        "--clip-timestamps", clip_timestamps_arg,
        "--initial-prompt", config.interview.vocab_prompt,
        "--condition-on-previous-text", "False",
        "--verbose", "False",
    ]

    if model_type == "turbo":
        cmd.extend(["--word-timestamps", "True"])

    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        timeout=300,
    )

    if res.returncode != 0:
        raise RuntimeError(
            f"定向修复转写失败 (退出码 {res.returncode}):\n{res.stderr or res.stdout}"
        )

    expected_json = out_d / f"{output_name}.json"
    if not expected_json.exists():
        raise FileNotFoundError(f"定向修复产物不存在: {expected_json}")

    return {
        "output_name": output_name,
        "output_json": str(expected_json),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "integer_start": r_start,
        "integer_end": r_end,
    }
