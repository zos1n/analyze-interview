"""
流水线总控编排模块：将 Stage 0 至 Stage 8 串联为完整的全自动化/单步执行流程，
支持断点恢复、异常现场保留与审计日志记录。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from analyze_interview.audio import run_stage_2_audio_pipeline
from analyze_interview.config import AnalyzerConfig, load_config
from analyze_interview.environment import check_environment
from analyze_interview.indexer import build_question_index, calculate_merged_coverage
from analyze_interview.key_transcript import run_stage_6_key_transcript
from analyze_interview.pathology import (
    detect_chinese_hallucination_phrases,
    deterministic_merge_segments,
    run_pathology_check,
)
from analyze_interview.precheck import run_precheck
from analyze_interview.reviewer import generate_codex_analysis_prompt, generate_review_skeleton
from analyze_interview.state import (
    ExecutionStage,
    ExecutionState,
    get_default_output_dir,
)
from analyze_interview.transcribe import (
    run_repair_transcribe,
    run_small_transcribe,
    run_turbo_clip,
)
from analyze_interview.verifier import run_stage_8_verification


def compensate_turbo_coverage(
    turbo_segments: List[Dict[str, Any]],
    question_events: List[Dict[str, Any]],
    total_duration_sec: float,
    current_intervals: List[Tuple[float, float]],
    coverage_threshold: float = 0.40,
) -> Tuple[List[Tuple[float, float]], float, List[Dict[str, Any]]]:
    """Stage 5 问句覆盖率 ±20s 校验与补选区间生成算法：
    1. 过滤掉命中中文幻觉词的段落，获得有效段；
    2. 遍历问句事件 Q_i，检查是否存在有效段满足：
       Turbo.start <= Q_i.end + 20s 且 Turbo.end >= Q_i.start - 20s；
    3. 未覆盖问句按时间升序排序，相邻间隔 <= 120s 合并为一簇；
    4. 补选区间为：[max(0, Cluster.start - 10s), min(Duration, Cluster.end + 60s)]；
    5. 计算与现有区间的合并覆盖率。
    返回：(补选区间列表, 合并后总覆盖率, 未覆盖问句列表)
    """
    # 1. 过滤幻觉段
    hallucination_hits = detect_chinese_hallucination_phrases(turbo_segments)
    bad_indices = {h["segment_index"] for h in hallucination_hits}
    valid_segs = [s for idx, s in enumerate(turbo_segments) if idx not in bad_indices]

    # 2. 检查问句覆盖
    uncovered_questions: List[Dict[str, Any]] = []
    for q in question_events:
        q_start = float(q.get("start", 0.0))
        q_end = float(q.get("end", 0.0))

        # 满足 Turbo.start <= Q_i.end + 20s 且 Turbo.end >= Q_i.start - 20s
        is_covered = any(
            float(s.get("start", 0.0)) <= q_end + 20.0 and float(s.get("end", 0.0)) >= q_start - 20.0
            for s in valid_segs
        )
        if not is_covered:
            uncovered_questions.append(q)

    if not uncovered_questions:
        _, ratio, _ = calculate_merged_coverage(current_intervals, total_duration_sec)
        return [], ratio, []

    # 3. 未覆盖问句按时间升序排序
    sorted_uncovered = sorted(uncovered_questions, key=lambda x: float(x.get("start", 0.0)))

    # 4. 相邻间隔 <= 120s 聚类
    clusters: List[List[Dict[str, Any]]] = []
    curr_cluster = [sorted_uncovered[0]]
    for q in sorted_uncovered[1:]:
        prev_end = float(curr_cluster[-1].get("end", 0.0))
        curr_start = float(q.get("start", 0.0))
        if curr_start - prev_end <= 120.0:
            curr_cluster.append(q)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [q]
    if curr_cluster:
        clusters.append(curr_cluster)

    # 5. 补选区间生成：[max(0, Cluster.start - 10s), min(Duration, Cluster.end + 60s)]
    compensation_intervals: List[Tuple[float, float]] = []
    for c in clusters:
        c_start = float(c[0].get("start", 0.0))
        c_end = float(c[-1].get("end", 0.0))
        comp_a = max(0.0, c_start - 10.0)
        comp_b = min(total_duration_sec, c_end + 60.0)
        compensation_intervals.append((comp_a, comp_b))

    # 计算与现有区间合并后的总覆盖率
    all_intervals = current_intervals + compensation_intervals
    _, total_ratio, merged = calculate_merged_coverage(all_intervals, total_duration_sec)

    return compensation_intervals, total_ratio, uncovered_questions


class InterviewAnalysisPipeline:
    """面试录屏分析全流程执行引擎"""

    def __init__(
        self,
        video_path: str | Path,
        config: Optional[AnalyzerConfig] = None,
        output_dir: Optional[str | Path] = None,
        resume: bool = True,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ):
        self.video_path = Path(video_path).expanduser().resolve()
        if not self.video_path.exists():
            raise FileNotFoundError(f"原视频文件不存在: {video_path}")

        self.config = config or load_config()
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else get_default_output_dir(self.video_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = ExecutionState(output_dir=self.output_dir, video_path=self.video_path)
        self.state.venv_path = self.config.runtime.venv_path
        self.resume = resume
        self.on_progress = on_progress or (lambda stage, msg: None)

    def _notify(self, stage: str, message: str) -> None:
        self.state.log(message, stage=stage)
        self.on_progress(stage, message)

    def step_0_precheck(self) -> Dict[str, Any]:
        """Stage 0: 授权与预检（网络连通性矩阵、代理诊断与 SHA256 基线）"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_0_PRECHECK):
            self._notify("stage_0_precheck", "已完成 Stage 0 预检，复用既有基线数据")
            return self.state.video_baseline

        self._notify("stage_0_precheck", f"开始 Stage 0 预检: 计算 {self.video_path.name} SHA256 基线与网络诊断...")
        res = run_precheck(self.video_path, self.output_dir, self.config)

        self.state.video_baseline = res["baseline"]
        self.state.network_decisions = res["decisions"]
        self.state.mark_stage_complete(
            ExecutionStage.STAGE_0_PRECHECK,
            next_stage=ExecutionStage.STAGE_1_ENV_INSTALL,
        )
        return res

    def step_1_environment(self) -> Dict[str, Any]:
        """Stage 1: 环境依赖核验（FFmpeg、共享 venv、模型缓存）"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_1_ENV_INSTALL):
            self._notify("stage_1_env_install", "已完成 Stage 1 环境核验，跳过")
            return {"all_ready": True}

        self._notify("stage_1_env_install", "开始 Stage 1 环境核验...")
        env_info = check_environment(self.config)

        if not env_info["all_ready"]:
            missing = env_info["missing_items"]
            self._notify("stage_1_env_install", f"发现缺失依赖项 ({len(missing)} 项)，需确认安装")
            # 记录缺失，但不强制报错退出，便于交互式 CLI 处理
        else:
            self._notify("stage_1_env_install", "所有环境依赖均已就绪")
            self.state.mark_stage_complete(
                ExecutionStage.STAGE_1_ENV_INSTALL,
                next_stage=ExecutionStage.STAGE_2_AUDIO_EXTRACT,
            )

        return env_info

    def step_2_audio_extract(self) -> Dict[str, Any]:
        """Stage 2: 音轨提取与多轨/声道智能分析"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_2_AUDIO_EXTRACT):
            self._notify("stage_2_audio_extract", "已完成 Stage 2 音频提取，跳过")
            return self.state.artifacts

        self._notify("stage_2_audio_extract", "开始 Stage 2 音轨提取与能量剖面互补性分析...")
        res = run_stage_2_audio_pipeline(self.video_path, self.output_dir, self.config)

        self.state.audio_mode = "split_tracks" if res["is_split_tracks"] else "single_track"
        for name, p in res["artifacts"].items():
            self.state.record_artifact(name, p)

        self.state.mark_stage_complete(
            ExecutionStage.STAGE_2_AUDIO_EXTRACT,
            next_stage=ExecutionStage.STAGE_3_SMALL_TRANSCRIBE,
        )
        return res

    def step_3_small_transcribe(self) -> Dict[str, Any]:
        """Stage 3: Whisper Small 全量粗转（支持单轨与分轨双粗转）"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_3_SMALL_TRANSCRIBE):
            self._notify("stage_3_small_transcribe", "已完成 Stage 3 Small 粗转，跳过")
            return self.state.artifacts

        self._notify("stage_3_small_transcribe", "开始 Stage 3 Whisper Small 全量粗转...")
        env_vars = self.state.network_decisions.get("env_vars", {})

        if self.state.audio_mode == "split_tracks":
            asr_track1 = self.state.artifacts.get("audio_asr_track_1") or (self.output_dir / "audio-asr-track1.flac")
            asr_track2 = self.state.artifacts.get("audio_asr_track_2") or (self.output_dir / "audio-asr-track2.flac")

            self._notify("stage_3_small_transcribe", "执行 Track 1 (面试官) 粗转...")
            res1 = run_small_transcribe(
                audio_path=asr_track1,
                output_dir=self.output_dir,
                config=self.config,
                output_name="transcript-full-track1",
                env_vars=env_vars,
            )
            self.state.record_artifact("transcript_full_track1_json", res1["output_json"])
            self.state.record_artifact("transcript_full_track1_srt", res1["output_srt"])
            self.state.record_artifact("transcript_full_json", res1["output_json"])

            self._notify("stage_3_small_transcribe", "执行 Track 2 (候选人) 粗转...")
            res2 = run_small_transcribe(
                audio_path=asr_track2,
                output_dir=self.output_dir,
                config=self.config,
                output_name="transcript-full-track2",
                env_vars=env_vars,
            )
            self.state.record_artifact("transcript_full_track2_json", res2["output_json"])
            self.state.record_artifact("transcript_full_track2_srt", res2["output_srt"])
            res = {"track1": res1, "track2": res2}
        else:
            asr_flac = self.state.artifacts.get("audio_asr") or (self.output_dir / "audio-asr.flac")
            res = run_small_transcribe(
                audio_path=asr_flac,
                output_dir=self.output_dir,
                config=self.config,
                output_name="transcript-full",
                env_vars=env_vars,
            )
            self.state.record_artifact("transcript_full_json", res["output_json"])
            self.state.record_artifact("transcript_full_srt", res["output_srt"])

        self.state.mark_stage_complete(
            ExecutionStage.STAGE_3_SMALL_TRANSCRIBE,
            next_stage=ExecutionStage.STAGE_3_5_PATHOLOGY_REPAIR,
        )
        return res

    def _check_and_repair_track(
        self,
        transcript_json_path: Path | str,
        asr_flac_path: Path | str,
        merged_filename: str,
        artifact_key: str,
        env_vars: Dict[str, str],
    ) -> Dict[str, Any]:
        """单轨病理检测与定向修复辅助函数"""
        transcript_json = Path(transcript_json_path).resolve()
        asr_flac = Path(asr_flac_path).resolve()

        patho_res = run_pathology_check(
            transcript_json_path=transcript_json,
            asr_flac_path=asr_flac,
            config=self.config,
            ffmpeg_path=self.config.runtime.ffmpeg,
        )

        merged_json_path = self.output_dir / merged_filename
        if patho_res.get("needs_repair"):
            self._notify("stage_3_5_pathology_repair", f"检测到 {transcript_json.name} 包含循环坏区间，执行定向重转与确定性合并...")
            with open(transcript_json, "r", encoding="utf-8") as f:
                orig_data = json.load(f)
            orig_segs = orig_data.get("segments", [])

            for loop in patho_res.get("repetition_loops", []):
                repair_start = max(0.0, loop["start"] - 10.0)
                repair_end = loop["end"]
                rep_res = run_repair_transcribe(
                    audio_path=asr_flac,
                    output_dir=self.output_dir,
                    model_type="small",
                    start_sec=repair_start,
                    end_sec=repair_end,
                    config=self.config,
                    env_vars=env_vars,
                )
                with open(rep_res["output_json"], "r", encoding="utf-8") as f:
                    rep_data = json.load(f)
                rep_segs = rep_data.get("segments", [])

                orig_segs = deterministic_merge_segments(
                    original_segments=orig_segs,
                    repair_segments=rep_segs,
                    repair_start_sec=repair_start,
                    repair_end_sec=repair_end,
                )

            with open(merged_json_path, "w", encoding="utf-8") as f:
                json.dump({"segments": orig_segs, "text": "".join(s.get("text", "") for s in orig_segs)}, f, indent=2, ensure_ascii=False)
        else:
            shutil_copy = Path(transcript_json).read_text(encoding="utf-8")
            merged_json_path.write_text(shutil_copy, encoding="utf-8")

        self.state.record_artifact(artifact_key, str(merged_json_path))
        return patho_res

    def step_3_5_pathology_repair(self) -> Dict[str, Any]:
        """Stage 3.5: 转写病理检测与定向修复/确定性合并（支持分轨双检双修）"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_3_5_PATHOLOGY_REPAIR):
            self._notify("stage_3_5_pathology_repair", "已完成 Stage 3.5 病理检测，跳过")
            return self.state.artifacts

        self._notify("stage_3_5_pathology_repair", "开始 Stage 3.5 转写病理检测 (静音幻觉/循环/断档)...")
        env_vars = self.state.network_decisions.get("env_vars", {})

        if self.state.audio_mode == "split_tracks":
            t1_json = self.state.artifacts.get("transcript_full_track1_json") or (self.output_dir / "transcript-full-track1.json")
            asr1_flac = self.state.artifacts.get("audio_asr_track_1") or (self.output_dir / "audio-asr-track1.flac")
            patho_res1 = self._check_and_repair_track(
                t1_json, asr1_flac, "transcript-full-track1-merged.json", "transcript_full_track1_merged_json", env_vars
            )
            self.state.record_artifact("transcript_full_merged_json", self.state.artifacts["transcript_full_track1_merged_json"])

            t2_json = self.state.artifacts.get("transcript_full_track2_json") or (self.output_dir / "transcript-full-track2.json")
            asr2_flac = self.state.artifacts.get("audio_asr_track_2") or (self.output_dir / "audio-asr-track2.flac")
            patho_res2 = self._check_and_repair_track(
                t2_json, asr2_flac, "transcript-full-track2-merged.json", "transcript_full_track2_merged_json", env_vars
            )
            patho_res = {"track1": patho_res1, "track2": patho_res2}
        else:
            transcript_json = self.state.artifacts.get("transcript_full_json") or (self.output_dir / "transcript-full.json")
            asr_flac = self.state.artifacts.get("audio_asr") or (self.output_dir / "audio-asr.flac")
            patho_res = self._check_and_repair_track(
                transcript_json, asr_flac, "transcript-full-merged.json", "transcript_full_merged_json", env_vars
            )

        self.state.mark_stage_complete(
            ExecutionStage.STAGE_3_5_PATHOLOGY_REPAIR,
            next_stage=ExecutionStage.STAGE_4_QUESTION_INDEX,
        )
        return patho_res

    def step_4_question_index(self) -> Dict[str, Any]:
        """Stage 4: 问题索引与密度聚类"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_4_QUESTION_INDEX):
            self._notify("stage_4_question_index", "已完成 Stage 4 问题索引，跳过")
            return self.state.artifacts

        self._notify("stage_4_question_index", "开始 Stage 4 问题索引与密度聚类 (生成 question-index.md)...")
        if self.state.audio_mode == "split_tracks":
            transcript_json = self.state.artifacts.get("transcript_full_track1_merged_json") or (self.output_dir / "transcript-full-track1-merged.json")
        else:
            transcript_json = self.state.artifacts.get("transcript_full_merged_json") or (self.output_dir / "transcript-full-merged.json")

        res = build_question_index(transcript_json, self.output_dir, self.config)

        self.state.record_artifact("question_index_md", res["question_index_path"])
        self.state.mark_stage_complete(
            ExecutionStage.STAGE_4_QUESTION_INDEX,
            next_stage=ExecutionStage.STAGE_5_TURBO_TRANSCRIBE,
        )
        return res

    def step_5_turbo_transcribe(self, intervals: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """Stage 5: Whisper Large V3 Turbo 局部精转（含分轨支持与 ±20s 自动补录机制）"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_5_TURBO_TRANSCRIBE):
            self._notify("stage_5_turbo_transcribe", "已完成 Stage 5 Turbo 精转，跳过")
            return self.state.artifacts

        self._notify("stage_5_turbo_transcribe", "开始 Stage 5 Whisper Large V3 Turbo 精转...")
        env_vars = self.state.network_decisions.get("env_vars", {})

        # 获取音频路径与待精转区间
        if self.state.audio_mode == "split_tracks":
            primary_asr = self.state.artifacts.get("audio_asr_track_1") or (self.output_dir / "audio-asr-track1.flac")
            transcript_full = self.state.artifacts.get("transcript_full_track1_merged_json") or (self.output_dir / "transcript-full-track1-merged.json")
        else:
            primary_asr = self.state.artifacts.get("audio_asr") or (self.output_dir / "audio-asr.flac")
            transcript_full = self.state.artifacts.get("transcript_full_merged_json") or (self.output_dir / "transcript-full-merged.json")

        index_data = build_question_index(transcript_full, self.output_dir, self.config)
        if not intervals:
            intervals = index_data.get("merged_intervals", [])

        # 1. 基础区间单 clip 独立精转 (Track 1 或单轨)
        turbo_clips: List[Dict[str, Any]] = []
        for idx, (a, b) in enumerate(intervals, 1):
            self._notify("stage_5_turbo_transcribe", f"精转区间 {idx}/{len(intervals)}: [{a:.1f}s - {b:.1f}s]")
            clip_res = run_turbo_clip(
                audio_path=primary_asr,
                output_dir=self.output_dir,
                clip_index=idx,
                start_sec=a,
                end_sec=b,
                config=self.config,
                env_vars=env_vars,
            )
            turbo_clips.append(clip_res)

        # 合并所有基础 Turbo Clip
        all_turbo_segs: List[Dict[str, Any]] = []
        for clip in turbo_clips:
            with open(clip["output_json"], "r", encoding="utf-8") as f:
                c_data = json.load(f)
                all_turbo_segs.extend(c_data.get("segments", []))

        all_turbo_segs.sort(key=lambda x: float(x.get("start", 0.0)))
        for i, s in enumerate(all_turbo_segs):
            s["id"] = i

        turbo_merged_path = self.output_dir / "transcript-turbo-merged.json"
        with open(turbo_merged_path, "w", encoding="utf-8") as f:
            json.dump({"segments": all_turbo_segs, "text": "".join(s.get("text", "") for s in all_turbo_segs)}, f, indent=2, ensure_ascii=False)
        self.state.record_artifact("transcript_turbo_merged_json", str(turbo_merged_path))

        # 2. 如果是分轨模式，对 Track 2 (候选人) 进行全量/分块 Turbo 精转
        if self.state.audio_mode == "split_tracks":
            self._notify("stage_5_turbo_transcribe", "分轨模式：对 Track 2 (候选人) 执行 Turbo 精转...")
            cand_asr = self.state.artifacts.get("audio_asr_track_2") or (self.output_dir / "audio-asr-track2.flac")
            total_dur = float(index_data.get("total_duration_sec", 0.0))

            cand_clips: List[Dict[str, Any]] = []
            chunk_size = 1800.0
            num_chunks = max(1, int(np.ceil(total_dur / chunk_size))) if total_dur > 0 else 1
            for c_idx in range(num_chunks):
                c_start = c_idx * chunk_size
                c_end = min(total_dur, (c_idx + 1) * chunk_size) if total_dur > 0 else 1800.0
                c_res = run_turbo_clip(
                    audio_path=cand_asr,
                    output_dir=self.output_dir,
                    clip_index=100 + c_idx + 1,
                    start_sec=c_start,
                    end_sec=c_end,
                    config=self.config,
                    env_vars=env_vars,
                )
                cand_clips.append(c_res)

            cand_turbo_segs: List[Dict[str, Any]] = []
            for cc in cand_clips:
                with open(cc["output_json"], "r", encoding="utf-8") as f:
                    cc_data = json.load(f)
                    cand_turbo_segs.extend(cc_data.get("segments", []))
            cand_turbo_segs.sort(key=lambda x: float(x.get("start", 0.0)))
            for i, s in enumerate(cand_turbo_segs):
                s["id"] = i

            cand_turbo_merged = self.output_dir / "transcript-turbo-track2-merged.json"
            with open(cand_turbo_merged, "w", encoding="utf-8") as f:
                json.dump({"segments": cand_turbo_segs, "text": "".join(s.get("text", "") for s in cand_turbo_segs)}, f, indent=2, ensure_ascii=False)
            self.state.record_artifact("transcript_turbo_track2_merged_json", str(cand_turbo_merged))

        # 3. Stage 5 问句覆盖率 ±20s 自动校验与补录机制
        q_events = index_data.get("question_events", [])
        total_dur = float(index_data.get("total_duration_sec", 0.0))
        comp_intervals, total_ratio, uncovered = compensate_turbo_coverage(
            turbo_segments=all_turbo_segs,
            question_events=q_events,
            total_duration_sec=total_dur,
            current_intervals=intervals,
            coverage_threshold=self.config.thresholds.coverage_auto,
        )

        final_merged_path = turbo_merged_path
        if comp_intervals and total_ratio <= self.config.thresholds.coverage_auto:
            self._notify(
                "stage_5_turbo_transcribe",
                f"检测到 {len(uncovered)} 个问句未在 Turbo 覆盖内，自动补转 {len(comp_intervals)} 个区间（补录后总覆盖率 {total_ratio:.1%} ≤ 40%）..."
            )
            comp_clips: List[Dict[str, Any]] = []
            next_clip_idx = len(turbo_clips) + 1
            for comp_a, comp_b in comp_intervals:
                clip_res = run_turbo_clip(
                    audio_path=primary_asr,
                    output_dir=self.output_dir,
                    clip_index=next_clip_idx,
                    start_sec=comp_a,
                    end_sec=comp_b,
                    config=self.config,
                    env_vars=env_vars,
                )
                comp_clips.append(clip_res)
                next_clip_idx += 1

            comp_segs: List[Dict[str, Any]] = []
            for clip in comp_clips:
                with open(clip["output_json"], "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                    comp_segs.extend(c_data.get("segments", []))

            v2_segs = list(all_turbo_segs)
            for c_seg in comp_segs:
                v2_segs.append(c_seg)
            v2_segs.sort(key=lambda x: float(x.get("start", 0.0)))
            for i, s in enumerate(v2_segs):
                s["id"] = i

            turbo_v2_path = self.output_dir / "transcript-turbo-merged-v2.json"
            with open(turbo_v2_path, "w", encoding="utf-8") as f:
                json.dump({"segments": v2_segs, "text": "".join(s.get("text", "") for s in v2_segs)}, f, indent=2, ensure_ascii=False)
            self.state.record_artifact("transcript_turbo_merged_json", str(turbo_v2_path))
            final_merged_path = turbo_v2_path

        self.state.mark_stage_complete(
            ExecutionStage.STAGE_5_TURBO_TRANSCRIBE,
            next_stage=ExecutionStage.STAGE_6_KEY_TRANSCRIPT,
        )
        return {
            "turbo_merged_json": str(final_merged_path),
            "clip_count": len(turbo_clips),
            "coverage_ratio": total_ratio,
        }

    def step_6_key_transcript(self) -> Dict[str, Any]:
        """Stage 6: 关键问答稿提炼 (生成 transcript-key.md)"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_6_KEY_TRANSCRIPT):
            self._notify("stage_6_key_transcript", "已完成 Stage 6 关键问答提炼，跳过")
            return self.state.artifacts

        self._notify("stage_6_key_transcript", "开始 Stage 6 关键问答稿提炼 (生成 transcript-key.md)...")
        turbo_json = self.state.artifacts.get("transcript_turbo_merged_json") or (self.output_dir / "transcript-turbo-merged.json")
        if not Path(turbo_json).exists():
            turbo_json = self.state.artifacts.get("transcript_full_merged_json") or (self.output_dir / "transcript-full-merged.json")

        index_res = build_question_index(turbo_json, self.output_dir, self.config)
        is_split = self.state.audio_mode == "split_tracks"

        candidate_track_json = None
        if is_split:
            candidate_track_json = (
                self.state.artifacts.get("transcript_turbo_track2_merged_json")
                or self.state.artifacts.get("transcript_full_track2_merged_json")
                or (self.output_dir / "transcript-turbo-track2-merged.json")
                or (self.output_dir / "transcript-full-track2-merged.json")
            )

        res = run_stage_6_key_transcript(
            merged_transcript_json_path=turbo_json,
            question_index_data=index_res,
            output_dir=self.output_dir,
            config=self.config,
            is_split_tracks=is_split,
            candidate_track_json_path=candidate_track_json,
        )

        self.state.record_artifact("transcript_key_md", res["transcript_key_path"])
        self.state.mark_stage_complete(
            ExecutionStage.STAGE_6_KEY_TRANSCRIPT,
            next_stage=ExecutionStage.STAGE_7_REVIEW,
        )
        return res

    def step_7_review(self, mode: str = "codex") -> Dict[str, Any]:
        """Stage 7: 逐题深度复盘（生成 codex 提示词或 review 模板骨架）"""
        if self.resume and self.state.is_stage_completed(ExecutionStage.STAGE_7_REVIEW):
            self._notify("stage_7_review", "已完成 Stage 7 复盘指令生成，跳过")
            return self.state.artifacts

        self._notify("stage_7_review", f"开始 Stage 7 逐题复盘准备 (模式: {mode})...")
        key_md = self.state.artifacts.get("transcript_key_md") or (self.output_dir / "transcript-key.md")

        # 生成 codex-analysis-prompt.md
        prompt_path = self.output_dir / "codex-analysis-prompt.md"
        generate_codex_analysis_prompt(key_md, prompt_path)
        self.state.record_artifact("codex_analysis_prompt_md", str(prompt_path))

        # 生成 interview-review.md 模板骨架
        review_path = self.output_dir / "interview-review.md"
        generate_review_skeleton([], review_path)
        self.state.record_artifact("interview_review_md", str(review_path))

        self.state.mark_stage_complete(
            ExecutionStage.STAGE_7_REVIEW,
            next_stage=ExecutionStage.STAGE_8_FINAL_VERIFY,
        )
        return {
            "prompt_path": str(prompt_path),
            "review_path": str(review_path),
        }

    def step_8_verify(self) -> Dict[str, Any]:
        """Stage 8: 最终验收核验"""
        self._notify("stage_8_final_verify", "开始 Stage 8 最终验收 (核对源视频 SHA256 与产物完整性)...")
        res = run_stage_8_verification(self.state)

        if res["passed"]:
            self.state.mark_stage_complete(ExecutionStage.STAGE_8_FINAL_VERIFY, next_stage=ExecutionStage.COMPLETED)
        else:
            self.state.record_error(ExecutionStage.STAGE_8_FINAL_VERIFY, f"验收未通过: {'; '.join(res['errors'])}")

        return res

    def run_all(self) -> Dict[str, Any]:
        """按顺序完整执行 Stage 0 至 Stage 8"""
        self.step_0_precheck()
        env_res = self.step_1_environment()
        if not env_res.get("all_ready", True):
            return {"status": "paused_waiting_environment", "missing_items": env_res.get("missing_items", [])}

        self.step_2_audio_extract()
        self.step_3_small_transcribe()
        self.step_3_5_pathology_repair()
        idx_res = self.step_4_question_index()

        # 检查覆盖率
        if idx_res.get("needs_user_menu"):
            self._notify("stage_4_question_index", "覆盖率超 40%，按规范出三选一菜单，暂停全自动流水线")
            return {
                "status": "paused_waiting_user_choice",
                "coverage_ratio": idx_res.get("coverage_ratio"),
                "question_index_path": idx_res.get("question_index_path"),
            }

        self.step_5_turbo_transcribe()
        self.step_6_key_transcript()
        self.step_7_review(mode="codex")
        verify_res = self.step_8_verify()

        return {
            "status": "completed" if verify_res["passed"] else "failed_verification",
            "artifacts": self.state.artifacts,
            "verification": verify_res,
        }
