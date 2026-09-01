"""
Stage 2 音频提取与声道/多轨角色智能分析模块。
使用 FFprobe 探测元数据，并通过原始 PCM 1s 逐块 RMS 计算能量剖面，
智能识别分轨录制（系统声音 vs 麦克风）与双声道独立性。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from analyze_interview.config import AnalyzerConfig


def probe_media_info(video_path: str | Path, ffprobe_path: str = "ffprobe") -> Dict[str, Any]:
    """使用 FFprobe 读取视频和全部音轨元数据"""
    v_path = Path(video_path).expanduser().resolve()
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(v_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if res.returncode != 0:
        raise RuntimeError(f"FFprobe 探测媒体信息失败: {res.stderr}")

    data = json.loads(res.stdout)
    format_info = data.get("format", {})
    duration_sec = float(format_info.get("duration", 0.0))

    audio_streams: List[Dict[str, Any]] = []
    audio_idx = 0
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio":
            disposition = s.get("disposition", {})
            audio_streams.append({
                "audio_track_index": audio_idx,
                "stream_index": s.get("index"),
                "codec_name": s.get("codec_name"),
                "sample_rate": int(s.get("sample_rate", 0)),
                "channels": int(s.get("channels", 0)),
                "channel_layout": s.get("channel_layout", "unknown"),
                "is_default": bool(disposition.get("default", 0) == 1),
                "duration_sec": float(s.get("duration", duration_sec)),
                "tags": s.get("tags", {}),
            })
            audio_idx += 1

    return {
        "video_path": str(v_path),
        "format_name": format_info.get("format_name"),
        "total_duration_sec": duration_sec,
        "total_size_bytes": int(format_info.get("size", 0)),
        "audio_stream_count": len(audio_streams),
        "audio_streams": audio_streams,
    }


def extract_audio_track(
    video_path: str | Path,
    output_flac_path: str | Path,
    stream_index: Optional[int] = None,
    as_asr_mono_16k: bool = False,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """从源视频提取指定音轨为 FLAC 格式"""
    v_path = Path(video_path).expanduser().resolve()
    out_p = Path(output_flac_path).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg_path, "-y", "-nostdin", "-i", str(v_path)]
    if stream_index is not None:
        cmd.extend(["-map", f"0:a:{stream_index}"])
    else:
        cmd.extend(["-map", "0:a:0?"])

    if as_asr_mono_16k:
        cmd.extend(["-ac", "1", "-ar", "16000"])

    cmd.extend(["-c:a", "flac", str(out_p)])

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg 音频提取失败: {res.stderr}")


def load_pcm_f32le_16k(flac_path: str | Path, channels: int = 1, ffmpeg_path: str = "ffmpeg") -> np.ndarray:
    """通过 FFmpeg 将音频无损解码为 16kHz float32 NumPy PCM 数组"""
    p = Path(flac_path).expanduser().resolve()
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-v", "error",
        "-i", str(p),
        "-f", "f32le",
        "-ac", str(channels),
        "-ar", "16000",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_bytes, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 解码 PCM 失败: {err.decode('utf-8', errors='ignore')}")

    pcm = np.frombuffer(raw_bytes, dtype=np.float32)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    return pcm


def compute_1s_rms_profile(
    flac_path: str | Path,
    ffmpeg_path: str = "ffmpeg",
    sample_rate: int = 16000,
) -> np.ndarray:
    """计算音频全片 1 秒逐块 RMS 均方根能量剖面"""
    pcm = load_pcm_f32le_16k(flac_path, channels=1, ffmpeg_path=ffmpeg_path)
    total_samples = len(pcm)
    if total_samples == 0:
        return np.array([], dtype=np.float32)

    total_seconds = int(np.ceil(total_samples / sample_rate))
    rms_profile = np.zeros(total_seconds, dtype=np.float32)

    for sec in range(total_seconds):
        start_idx = sec * sample_rate
        end_idx = min((sec + 1) * sample_rate, total_samples)
        if start_idx < end_idx:
            chunk = pcm[start_idx:end_idx]
            rms = np.sqrt(np.mean(chunk ** 2))
            rms_profile[sec] = rms

    return rms_profile


def analyze_multitrack_complementarity(
    track_profiles: List[np.ndarray],
    silence_threshold: float = 0.005,
) -> Dict[str, Any]:
    """分析多音轨之间的能量相关性与互补性，自动判定是否为分轨录制"""
    n_tracks = len(track_profiles)
    if n_tracks < 2:
        return {
            "mode": "single_track",
            "is_split_tracks": False,
            "correlation_matrix": [[1.0]],
            "notes": ["仅检测到单条音轨"],
        }

    # 对齐长度
    min_len = min(len(p) for p in track_profiles)
    truncated = [p[:min_len] for p in track_profiles]

    corr_matrix = np.corrcoef(truncated)
    # 处理 NaN
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 计算 Track 0 与 Track 1 之间的互补性
    p0 = truncated[0]
    p1 = truncated[1]

    t0_active = p0 >= silence_threshold
    t1_active = p1 >= silence_threshold

    # 互补秒数：一方有声，另一方静音
    t0_only = np.sum(t0_active & ~t1_active)
    t1_only = np.sum(~t0_active & t1_active)
    both_active = np.sum(t0_active & t1_active)
    total_active = np.sum(t0_active | t1_active)

    t0_only_ratio = (t0_only / total_active) if total_active > 0 else 0.0
    t1_only_ratio = (t1_only / total_active) if total_active > 0 else 0.0
    complementary_ratio = ((t0_only + t1_only) / total_active) if total_active > 0 else 0.0
    pair_corr = float(corr_matrix[0][1])

    # 判定规则：
    # 若互补占比高 (> 25%) 且相关系数低 (< 0.6) -> 分轨模式 (系统声音 + 麦克风)
    # 若相关系数 ≈ 1.0 且互补占比低 -> 冗余轨
    if complementary_ratio > 0.25 and pair_corr < 0.6:
        mode = "split_tracks"
        is_split = True
        notes = [
            f"检测到明显分轨模式（互补能量占比 {complementary_ratio:.1%}，相关系数 {pair_corr:.2f}）",
            f"音轨 0 单独发声: {t0_only}s ({t0_only_ratio:.1%})，音轨 1 单独发声: {t1_only}s ({t1_only_ratio:.1%})",
            "决策：双轨分别导出转写，按时间戳合并，避免漏转候选人或面试官",
        ]
    elif pair_corr > 0.9:
        mode = "redundant_tracks"
        is_split = False
        notes = [
            f"检测到冗余音轨（相关系数 {pair_corr:.2f} ≈ 1.0），取第一轨处理",
        ]
    else:
        mode = "multiple_tracks_mixed"
        is_split = False
        notes = [
            f"多音轨相关系数 {pair_corr:.2f}，互补占比 {complementary_ratio:.1%}，默认使用首轨",
        ]

    return {
        "mode": mode,
        "is_split_tracks": is_split,
        "correlation": pair_corr,
        "complementary_ratio": complementary_ratio,
        "t0_only_seconds": int(t0_only),
        "t1_only_seconds": int(t1_only),
        "both_active_seconds": int(both_active),
        "notes": notes,
    }


def analyze_stereo_channel_independence(
    flac_path: str | Path,
    ffmpeg_path: str = "ffmpeg",
) -> Dict[str, Any]:
    """分析立体声左右声道的独立性（计算左右通道 Pearson 相关系数与能量差异）"""
    pcm_stereo = load_pcm_f32le_16k(flac_path, channels=2, ffmpeg_path=ffmpeg_path)
    if len(pcm_stereo) == 0:
        return {"independent": False, "correlation": 1.0, "diff_energy_ratio": 0.0}

    left = pcm_stereo[:, 0]
    right = pcm_stereo[:, 1]

    # Pearson 相关系数
    corr = float(np.corrcoef(left, right)[0, 1])
    if np.isnan(corr):
        corr = 1.0

    left_energy = float(np.sum(left ** 2))
    right_energy = float(np.sum(right ** 2))
    diff_energy = float(np.sum((left - right) ** 2))
    total_energy = max(left_energy + right_energy, 1e-9)
    diff_ratio = diff_energy / total_energy

    # 相关系数 < 0.85 且差异能量占比 > 0.15 判定为独立左右声道
    independent = (corr < 0.85) and (diff_ratio > 0.15)

    return {
        "independent": independent,
        "correlation": round(corr, 4),
        "diff_energy_ratio": round(diff_ratio, 4),
        "left_energy": round(left_energy, 2),
        "right_energy": round(right_energy, 2),
    }


def validate_audio_integrity(
    audio_path: str | Path,
    expected_duration_sec: float,
    ffmpeg_path: str = "ffmpeg",
) -> Dict[str, Any]:
    """验收导出的音频文件完整性（非空、无错误解码、时长匹配）"""
    p = Path(audio_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return {"passed": False, "error": f"音频文件不存在或为空: {audio_path}"}

    # ffmpeg -v error -f null -
    cmd = [ffmpeg_path, "-v", "error", "-i", str(p), "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        return {"passed": False, "error": f"音频解码失败: {res.stderr}"}

    # 检查时长
    # 读采样点推导时长
    pcm = load_pcm_f32le_16k(p, channels=1, ffmpeg_path=ffmpeg_path)
    actual_duration = len(pcm) / 16000.0
    duration_diff = abs(actual_duration - expected_duration_sec)

    warnings: List[str] = []
    if duration_diff > 2.0:
        if actual_duration < expected_duration_sec - 5.0:
            warnings.append(
                f"音频时长 ({actual_duration:.1f}s) 比视频 ({expected_duration_sec:.1f}s) 短超过 5s"
            )
        else:
            warnings.append(
                f"音频时长 ({actual_duration:.1f}s) 与视频 ({expected_duration_sec:.1f}s) 相差 {duration_diff:.1f}s"
            )

    return {
        "passed": True,
        "actual_duration_sec": round(actual_duration, 2),
        "expected_duration_sec": round(expected_duration_sec, 2),
        "duration_diff_sec": round(duration_diff, 2),
        "warnings": warnings,
    }


def run_stage_2_audio_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    config: AnalyzerConfig,
) -> Dict[str, Any]:
    """执行 Stage 2 完整音轨提取与分析流水线"""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = config.runtime.ffmpeg
    ffprobe_bin = config.runtime.ffprobe

    # 1. FFprobe 探测元信息
    media_info = probe_media_info(video_path, ffprobe_path=ffprobe_bin)
    audio_streams = media_info.get("audio_streams", [])
    if not audio_streams:
        raise RuntimeError("视频中未检测到任何有效音轨！")

    source_info_path = out_dir / "source-info.json"
    artifacts: Dict[str, str] = {"source_info": str(source_info_path)}

    # 2. 多音轨时做 1s RMS 能量互补性分析
    track_profiles: List[np.ndarray] = []
    temp_track_files: List[Path] = []

    for stream_meta in audio_streams:
        track_idx = stream_meta["audio_track_index"]
        temp_flac = out_dir / f"temp-track{track_idx}-16k.flac"
        extract_audio_track(
            video_path,
            temp_flac,
            stream_index=track_idx,
            as_asr_mono_16k=True,
            ffmpeg_path=ffmpeg_bin,
        )
        temp_track_files.append(temp_flac)
        profile = compute_1s_rms_profile(temp_flac, ffmpeg_path=ffmpeg_bin)
        track_profiles.append(profile)

    multitrack_analysis = analyze_multitrack_complementarity(
        track_profiles,
        silence_threshold=config.thresholds.silence_rms_threshold,
    )

    is_split = multitrack_analysis.get("is_split_tracks", False)

    if is_split:
        # 分轨模式：分别导出各轨的 audio-source-track<N>.flac 与 audio-asr-track<N>.flac
        for stream_meta in audio_streams:
            track_idx = stream_meta["audio_track_index"]
            src_track_flac = out_dir / f"audio-source-track{track_idx + 1}.flac"
            asr_track_flac = out_dir / f"audio-asr-track{track_idx + 1}.flac"

            extract_audio_track(
                video_path,
                src_track_flac,
                stream_index=track_idx,
                as_asr_mono_16k=False,
                ffmpeg_path=ffmpeg_bin,
            )
            # 重命名或生成 ASR FLAC
            temp_flac = temp_track_files[track_idx]
            if temp_flac.exists():
                temp_flac.rename(asr_track_flac)
            else:
                extract_audio_track(
                    video_path,
                    asr_track_flac,
                    stream_index=track_idx,
                    as_asr_mono_16k=True,
                    ffmpeg_path=ffmpeg_bin,
                )

            artifacts[f"audio_source_track_{track_idx + 1}"] = str(src_track_flac)
            artifacts[f"audio_asr_track_{track_idx + 1}"] = str(asr_track_flac)

        primary_asr_flac = out_dir / "audio-asr-track1.flac"
    else:
        # 单轨模式：导出 audio-source.flac 与 audio-asr.flac
        src_flac = out_dir / "audio-source.flac"
        asr_flac = out_dir / "audio-asr.flac"

        extract_audio_track(
            video_path,
            src_flac,
            stream_index=0,
            as_asr_mono_16k=False,
            ffmpeg_path=ffmpeg_bin,
        )
        temp_flac = temp_track_files[0]
        if temp_flac.exists():
            temp_flac.rename(asr_flac)
        else:
            extract_audio_track(
                video_path,
                asr_flac,
                stream_index=0,
                as_asr_mono_16k=True,
                ffmpeg_path=ffmpeg_bin,
            )

        artifacts["audio_source"] = str(src_flac)
        artifacts["audio_asr"] = str(asr_flac)
        primary_asr_flac = asr_flac

    # 清理其他临时文件
    for t_f in temp_track_files:
        if t_f.exists():
            t_f.unlink()

    # 3. 立体声独立性检查（针对主音频源）
    stereo_info = {}
    if audio_streams[0]["channels"] >= 2:
        source_for_stereo = artifacts.get("audio_source") or artifacts.get("audio_source_track_1")
        if source_for_stereo:
            stereo_info = analyze_stereo_channel_independence(source_for_stereo, ffmpeg_path=ffmpeg_bin)
            if stereo_info.get("independent"):
                left_flac = out_dir / "audio-left.flac"
                right_flac = out_dir / "audio-right.flac"
                # 左声道提取
                subprocess.run([
                    ffmpeg_bin, "-y", "-nostdin", "-i", str(source_for_stereo),
                    "-af", "pan=mono|c0=c0", "-c:a", "flac", str(left_flac),
                ], capture_output=True, timeout=120)
                # 右声道提取
                subprocess.run([
                    ffmpeg_bin, "-y", "-nostdin", "-i", str(source_for_stereo),
                    "-af", "pan=mono|c0=c1", "-c:a", "flac", str(right_flac),
                ], capture_output=True, timeout=120)
                artifacts["audio_left"] = str(left_flac)
                artifacts["audio_right"] = str(right_flac)

    # 4. 音频验收
    val_res = validate_audio_integrity(
        primary_asr_flac,
        expected_duration_sec=media_info["total_duration_sec"],
        ffmpeg_path=ffmpeg_bin,
    )
    if not val_res["passed"]:
        raise RuntimeError(f"音频完整性校验未通过: {val_res.get('error')}")

    # 保存 source-info.json
    source_info_data = {
        "media_info": media_info,
        "multitrack_analysis": multitrack_analysis,
        "stereo_analysis": stereo_info,
        "validation": val_res,
        "artifacts": artifacts,
    }
    with open(source_info_path, "w", encoding="utf-8") as f:
        json.dump(source_info_data, f, indent=2, ensure_ascii=False)

    return {
        "source_info": source_info_data,
        "is_split_tracks": is_split,
        "artifacts": artifacts,
    }
