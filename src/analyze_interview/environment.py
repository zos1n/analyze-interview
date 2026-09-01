"""
Stage 1 环境依赖核验模块：负责检查 FFmpeg、共享 venv (mlx-whisper 0.4.3) 与模型缓存状态。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyze_interview.config import AnalyzerConfig


def find_executable(configured_path: str, fallback_name: str) -> Optional[str]:
    """寻找可执行文件路径"""
    p = Path(configured_path).expanduser()
    if p.exists() and os.access(p, os.X_OK):
        return str(p)
    system_which = shutil.which(fallback_name)
    if system_which:
        return system_which
    return None


def get_command_version(cmd: List[str]) -> Optional[str]:
    """获取命令输出的版本信息第一行"""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            lines = res.stdout.splitlines() or res.stderr.splitlines()
            return lines[0].strip() if lines else "unknown"
    except Exception:
        pass
    return None


def check_shared_venv(venv_path_str: str, expected_mlx_whisper_version: str = "0.4.3") -> Dict[str, Any]:
    """检查共享 venv 是否存在且具备 mlx-whisper==0.4.3"""
    v_dir = Path(venv_path_str).expanduser()
    py_bin = v_dir / "bin" / "python"
    pip_bin = v_dir / "bin" / "pip"
    mlx_whisper_bin = v_dir / "bin" / "mlx_whisper"

    exists = v_dir.exists() and py_bin.exists()
    mlx_version: Optional[str] = None
    mlx_whisper_version: Optional[str] = None
    version_matched = False

    if exists:
        # 使用 venv python 检查安装包
        try:
            check_script = (
                "import mlx_whisper, mlx.core, sys; "
                "print(getattr(mlx_whisper, '__version__', 'unknown')); "
                "print(getattr(mlx.core, '__version__', 'unknown'))"
            )
            res = subprocess.run(
                [str(py_bin), "-c", check_script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                lines = res.stdout.strip().splitlines()
                if len(lines) >= 1:
                    mlx_whisper_version = lines[0].strip()
                if len(lines) >= 2:
                    mlx_version = lines[1].strip()
                version_matched = (mlx_whisper_version == expected_mlx_whisper_version)
        except Exception:
            pass

    return {
        "venv_path": str(v_dir),
        "exists": exists,
        "python_bin": str(py_bin) if py_bin.exists() else None,
        "pip_bin": str(pip_bin) if pip_bin.exists() else None,
        "mlx_whisper_bin": str(mlx_whisper_bin) if mlx_whisper_bin.exists() else None,
        "mlx_whisper_version": mlx_whisper_version,
        "mlx_version": mlx_version,
        "expected_mlx_whisper_version": expected_mlx_whisper_version,
        "version_matched": version_matched,
    }


def check_hf_model_cache(model_id: str) -> Dict[str, Any]:
    """检查 Hugging Face 本地缓存目录中是否存在指定模型"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    folder_name = "models--" + model_id.replace("/", "--")
    model_path = cache_dir / folder_name
    cached = False
    cached_revisions: List[str] = []

    if model_path.exists() and model_path.is_dir():
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            for snap in snapshots_dir.iterdir():
                if snap.is_dir():
                    cached_revisions.append(snap.name)
            if cached_revisions:
                cached = True

    return {
        "model_id": model_id,
        "cached": cached,
        "cache_path": str(model_path),
        "cached_revisions": cached_revisions,
    }


def check_environment(config: AnalyzerConfig) -> Dict[str, Any]:
    """核验系统环境、FFmpeg、共享 venv 与模型缓存"""
    ffmpeg_path = find_executable(config.runtime.ffmpeg, "ffmpeg")
    ffprobe_path = find_executable(config.runtime.ffprobe, "ffprobe")
    ffmpeg_version = get_command_version([ffmpeg_path, "-version"]) if ffmpeg_path else None
    ffprobe_version = get_command_version([ffprobe_path, "-version"]) if ffprobe_path else None

    venv_info = check_shared_venv(
        config.runtime.venv_path,
        expected_mlx_whisper_version=config.models.mlx_whisper,
    )

    small_cache = check_hf_model_cache(config.models.small)
    turbo_cache = check_hf_model_cache(config.models.turbo)

    missing_items: List[Dict[str, str]] = []

    if not ffmpeg_path or not ffprobe_path:
        missing_items.append({
            "item": "FFmpeg / FFprobe",
            "command": "/opt/homebrew/bin/brew install ffmpeg",
            "download_content": "Homebrew 官方 bottles",
            "estimated_disk": "0.5 - 1.5GB",
        })

    if not venv_info["exists"] or not venv_info["version_matched"]:
        venv_p = config.runtime.venv_path
        venv_parent = str(Path(venv_p).parent)
        pip_cmd = (
            f"mkdir -p {venv_parent} && python3 -m venv {venv_p} && "
            f"{venv_p}/bin/pip install 'mlx-whisper=={config.models.mlx_whisper}'"
        )
        missing_items.append({
            "item": f"共享 venv (mlx-whisper=={config.models.mlx_whisper})",
            "command": pip_cmd,
            "download_content": "PyPI (或清华镜像)",
            "estimated_disk": "~760MB",
        })

    if not small_cache["cached"]:
        missing_items.append({
            "item": f"Whisper Small 模型 ({config.models.small})",
            "command": "转写阶段首次调用自动拉取",
            "download_content": "Hugging Face Hub",
            "estimated_disk": "~500MB",
        })

    if not turbo_cache["cached"]:
        missing_items.append({
            "item": f"Whisper Turbo 模型 ({config.models.turbo})",
            "command": "精转阶段首次调用自动拉取",
            "download_content": "Hugging Face Hub",
            "estimated_disk": "~1.6GB",
        })

    env_summary = {
        "os": platform.system(),
        "arch": platform.machine(),
        "macos_version": platform.mac_ver()[0] if hasattr(platform, "mac_ver") else "unknown",
        "python_system": platform.python_version(),
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_path": ffprobe_path,
        "ffprobe_version": ffprobe_version,
        "venv_info": venv_info,
        "small_model_cache": small_cache,
        "turbo_model_cache": turbo_cache,
        "missing_items": missing_items,
        "all_ready": len(missing_items) == 0,
    }
    return env_summary
