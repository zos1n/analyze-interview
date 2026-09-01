"""
状态与审计日志管理模块：维护 execution-state.json 和 execution-log.txt。
支持全流程阶段推进、断点恢复与异常保留。
"""

from __future__ import annotations

import datetime
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExecutionStage(str, Enum):
    STAGE_0_PRECHECK = "stage_0_precheck"
    STAGE_1_ENV_INSTALL = "stage_1_env_install"
    STAGE_2_AUDIO_EXTRACT = "stage_2_audio_extract"
    STAGE_3_SMALL_TRANSCRIBE = "stage_3_small_transcribe"
    STAGE_3_5_PATHOLOGY_REPAIR = "stage_3_5_pathology_repair"
    STAGE_4_QUESTION_INDEX = "stage_4_question_index"
    STAGE_5_TURBO_TRANSCRIBE = "stage_5_turbo_transcribe"
    STAGE_6_KEY_TRANSCRIPT = "stage_6_key_transcript"
    STAGE_7_REVIEW = "stage_7_review"
    STAGE_8_FINAL_VERIFY = "stage_8_final_verify"
    COMPLETED = "completed"
    FAILED = "failed"


STAGE_ORDER: List[ExecutionStage] = [
    ExecutionStage.STAGE_0_PRECHECK,
    ExecutionStage.STAGE_1_ENV_INSTALL,
    ExecutionStage.STAGE_2_AUDIO_EXTRACT,
    ExecutionStage.STAGE_3_SMALL_TRANSCRIBE,
    ExecutionStage.STAGE_3_5_PATHOLOGY_REPAIR,
    ExecutionStage.STAGE_4_QUESTION_INDEX,
    ExecutionStage.STAGE_5_TURBO_TRANSCRIBE,
    ExecutionStage.STAGE_6_KEY_TRANSCRIPT,
    ExecutionStage.STAGE_7_REVIEW,
    ExecutionStage.STAGE_8_FINAL_VERIFY,
]


class ExecutionState:
    """运行状态与日志管理器"""

    def __init__(self, output_dir: str | Path, video_path: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.video_path = str(Path(video_path).expanduser().resolve())
        self.state_file = self.output_dir / "execution-state.json"
        self.log_file = self.output_dir / "execution-log.txt"

        self.venv_path: str = ""
        self.video_baseline: Dict[str, Any] = {}
        self.selected_audio_track: Optional[int] = None
        self.audio_mode: str = "single_track"  # single_track | split_tracks | stereo_split
        self.current_stage: ExecutionStage = ExecutionStage.STAGE_0_PRECHECK
        self.completed_stages: List[str] = []
        self.completed_ranges: List[List[float]] = []
        self.small_model_revision: str = ""
        self.turbo_model_revision: str = ""
        self.artifacts: Dict[str, str] = {}
        self.network_decisions: Dict[str, Any] = {}
        self.last_error: Optional[str] = None
        self.created_at: str = datetime.datetime.now().isoformat()
        self.updated_at: str = self.created_at

        # 确保产物目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.state_file.exists():
            self.load()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "output_dir": str(self.output_dir),
            "venv_path": self.venv_path,
            "video_baseline": self.video_baseline,
            "selected_audio_track": self.selected_audio_track,
            "audio_mode": self.audio_mode,
            "current_stage": str(self.current_stage.value if isinstance(self.current_stage, ExecutionStage) else self.current_stage),
            "completed_stages": self.completed_stages,
            "completed_ranges": self.completed_ranges,
            "small_model_revision": self.small_model_revision,
            "turbo_model_revision": self.turbo_model_revision,
            "artifacts": self.artifacts,
            "network_decisions": self.network_decisions,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def load(self) -> None:
        """从 JSON 加载状态"""
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.video_path = data.get("video_path", self.video_path)
            self.venv_path = data.get("venv_path", "")
            self.video_baseline = data.get("video_baseline", {})
            self.selected_audio_track = data.get("selected_audio_track")
            self.audio_mode = data.get("audio_mode", "single_track")
            raw_stage = data.get("current_stage", ExecutionStage.STAGE_0_PRECHECK.value)
            try:
                self.current_stage = ExecutionStage(raw_stage)
            except ValueError:
                self.current_stage = ExecutionStage.STAGE_0_PRECHECK
            self.completed_stages = data.get("completed_stages", [])
            self.completed_ranges = data.get("completed_ranges", [])
            self.small_model_revision = data.get("small_model_revision", "")
            self.turbo_model_revision = data.get("turbo_model_revision", "")
            self.artifacts = data.get("artifacts", {})
            self.network_decisions = data.get("network_decisions", {})
            self.last_error = data.get("last_error")
            self.created_at = data.get("created_at", self.created_at)
            self.updated_at = data.get("updated_at", self.updated_at)
        except Exception as e:
            self.log(f"加载状态文件失败: {e}", level="WARN")

    def save(self) -> None:
        """持久化保存当前状态"""
        self.updated_at = datetime.datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def log(self, message: str, level: str = "INFO", stage: Optional[str] = None) -> None:
        """写入审计日志 (execution-log.txt)"""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stage_str = stage or (self.current_stage.value if isinstance(self.current_stage, ExecutionStage) else str(self.current_stage))
        log_line = f"[{now_str}] [{level.upper()}] [{stage_str}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

    def mark_stage_complete(self, stage: ExecutionStage, next_stage: Optional[ExecutionStage] = None) -> None:
        """标记阶段已完成并推进一步"""
        stage_val = stage.value if isinstance(stage, ExecutionStage) else str(stage)
        if stage_val not in self.completed_stages:
            self.completed_stages.append(stage_val)
        if next_stage:
            self.current_stage = next_stage
        self.last_error = None
        self.log(f"阶段完成: {stage_val}", level="INFO", stage=stage_val)
        self.save()

    def record_artifact(self, name: str, file_path: str | Path) -> None:
        """登记产物"""
        p = Path(file_path).resolve()
        self.artifacts[name] = str(p)
        self.log(f"登记产物 [{name}]: {p}", level="INFO")
        self.save()

    def record_error(self, stage: ExecutionStage, error_msg: str) -> None:
        """记录错误并挂起状态"""
        stage_val = stage.value if isinstance(stage, ExecutionStage) else str(stage)
        self.last_error = f"[{stage_val}] {error_msg}"
        self.log(f"执行失败: {error_msg}", level="ERROR", stage=stage_val)
        self.save()

    def is_stage_completed(self, stage: ExecutionStage) -> bool:
        """检查特定阶段是否已经完成"""
        stage_val = stage.value if isinstance(stage, ExecutionStage) else str(stage)
        return stage_val in self.completed_stages


def get_default_output_dir(video_path: str | Path) -> Path:
    """根据视频路径自动推导产物目录: <视频所在目录>/<视频文件名去扩展名>_analysis/"""
    v_path = Path(video_path).expanduser().resolve()
    parent_dir = v_path.parent
    stem = v_path.stem
    return parent_dir / f"{stem}_analysis"
