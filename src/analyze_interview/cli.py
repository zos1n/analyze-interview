"""
统一命令行入口模块：提供丰富的子命令与交互式终端展现 (Rich UI)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from analyze_interview import __version__
from analyze_interview.config import load_config
from analyze_interview.pipeline import InterviewAnalysisPipeline
from analyze_interview.state import ExecutionState, get_default_output_dir

console = Console()


def _print_banner() -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]面试录屏分析工具 (analyze-interview) v{__version__}[/bold cyan]\n"
            "[dim]全本地 · 零云端费用 · 隐私保护 · Apple Silicon MLX 加速[/dim]",
            border_style="cyan",
        )
    )


def cmd_run(args: argparse.Namespace) -> int:
    """运行完整或单步分析流水线"""
    _print_banner()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        console.print(f"[bold red]错误：视频文件不存在: {video_path}[/bold red]")
        return 1

    config = load_config(args.config)
    output_dir = Path(args.output).expanduser().resolve() if args.output else get_default_output_dir(video_path)

    console.print(f"[green]目标视频:[/green] {video_path}")
    console.print(f"[green]产物目录:[/green] {output_dir}\n")

    def progress_callback(stage: str, msg: str) -> None:
        console.print(f"[{stage}] {msg}")

    pipeline = InterviewAnalysisPipeline(
        video_path=video_path,
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume,
        on_progress=progress_callback,
    )

    if args.stage:
        stage_name = args.stage.lower()
        if stage_name in ["0", "precheck"]:
            pipeline.step_0_precheck()
        elif stage_name in ["1", "env", "environment"]:
            pipeline.step_1_environment()
        elif stage_name in ["2", "audio", "extract"]:
            pipeline.step_2_audio_extract()
        elif stage_name in ["3", "small", "rough"]:
            pipeline.step_3_small_transcribe()
        elif stage_name in ["3.5", "pathology"]:
            pipeline.step_3_5_pathology_repair()
        elif stage_name in ["4", "index"]:
            pipeline.step_4_question_index()
        elif stage_name in ["5", "turbo"]:
            pipeline.step_5_turbo_transcribe()
        elif stage_name in ["6", "key"]:
            pipeline.step_6_key_transcript()
        elif stage_name in ["7", "review"]:
            pipeline.step_7_review(mode=args.review_mode or "codex")
        elif stage_name in ["8", "verify"]:
            pipeline.step_8_verify()
        else:
            console.print(f"[bold red]未知的 stage: {args.stage}[/bold red]")
            return 1
    else:
        res = pipeline.run_all()
        status = res.get("status")
        if status == "paused_waiting_environment":
            console.print("\n[bold yellow]流水线暂停：存在缺失的环境依赖，请查看上方提示安装后继续。[/bold yellow]")
        elif status == "paused_waiting_user_choice":
            console.print("\n[bold yellow]流水线暂停：候选区间覆盖率超 40%，请查看 question-index.md 手选或指定选项。[/bold yellow]")
        elif status == "completed":
            console.print("\n[bold green]恭喜！面试录屏全流程分析与验收已圆满完成！[/bold green]")

    return 0


def cmd_precheck(args: argparse.Namespace) -> int:
    """仅运行 Stage 0 预检"""
    _print_banner()
    video_path = Path(args.video).expanduser().resolve()
    config = load_config(args.config)
    output_dir = get_default_output_dir(video_path)

    pipeline = InterviewAnalysisPipeline(video_path=video_path, config=config, output_dir=output_dir)
    res = pipeline.step_0_precheck()

    table = Table(title="Stage 0 预检诊断结果", border_style="cyan")
    table.add_column("检查项", style="bold")
    table.add_column("状态 / 详情")

    base = res.get("baseline", {})
    table.add_row("视频文件", f"{base.get('video_path')} ({base.get('size_mb')} MB)")
    table.add_row("SHA-256 基线", f"{base.get('sha256')}")

    disk = res.get("disk", {})
    table.add_row("磁盘剩余空间", f"{disk.get('free_gb')} GB (要求 ≥ {disk.get('min_required_gb')} GB)")

    decisions = res.get("decisions", {})
    table.add_row("PyPI 策略", decisions.get("pip_index_url", "direct"))
    table.add_row("HF 镜像", decisions.get("hf_endpoint") or "官方直连")

    console.print(table)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """查看现有产物目录的状态与断点记录"""
    _print_banner()
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else get_default_output_dir(video_path)

    state = ExecutionState(output_dir=output_dir, video_path=video_path)

    table = Table(title=f"录屏分析断点状态: {output_dir.name}", border_style="green")
    table.add_column("属性", style="bold")
    table.add_column("值")

    table.add_row("当前阶段", str(state.current_stage.value if hasattr(state.current_stage, 'value') else state.current_stage))
    table.add_row("已完成阶段", ", ".join(state.completed_stages) if state.completed_stages else "无")
    table.add_row("音轨模式", state.audio_mode)
    table.add_row("最后错误", state.last_error or "无")
    table.add_row("更新时间", state.updated_at)

    console.print(table)

    if state.artifacts:
        art_table = Table(title="已登记产物列表", border_style="cyan")
        art_table.add_column("产物名称", style="bold yellow")
        art_table.add_column("绝对路径")
        for k, v in state.artifacts.items():
            art_table.add_row(k, v)
        console.print(art_table)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze-interview",
        description="面试录屏本地转写与深度复盘工具 (MLX Whisper ASR + Local Deep Review)",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 1. run
    p_run = subparsers.add_parser("run", help="执行分析流水线 (支持单步或全自动)")
    p_run.add_argument("--video", "-i", required=True, help="面试视频绝对路径")
    p_run.add_argument("--output", "-o", help="自定义产物目录 (默认: <视频目录>/<视频名>_analysis/)")
    p_run.add_argument("--stage", "-s", help="指定执行单个阶段 (0, 1, 2, 3, 3.5, 4, 5, 6, 7, 8)")
    p_run.add_argument("--config", "-c", help="自定义 YAML 配置文件路径")
    p_run.add_argument("--no-resume", action="store_true", help="强制从头执行，不使用既有断点")
    p_run.add_argument("--review-mode", choices=["codex", "api", "self"], default="codex", help="阶段 7 复盘模式")
    p_run.set_defaults(func=cmd_run)

    # 2. precheck
    p_precheck = subparsers.add_parser("precheck", help="仅执行 Stage 0 预检与基线计算")
    p_precheck.add_argument("--video", "-i", required=True, help="面试视频绝对路径")
    p_precheck.add_argument("--config", "-c", help="自定义 YAML 配置文件路径")
    p_precheck.set_defaults(func=cmd_precheck)

    # 3. status
    p_status = subparsers.add_parser("status", help="查看视频对应分析任务的状态与产物")
    p_status.add_argument("--video", "-i", required=True, help="面试视频绝对路径")
    p_status.add_argument("--output", "-o", help="自定义产物目录")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
