"""
配置模块：负责加载、合并与验证 analyzer-config.yaml。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """轻量级 YAML 回退解析器（用于未安装 pyyaml 的极简环境）"""
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_subsec: Optional[str] = None
    current_list: Optional[List[str]] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.strip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            result[current_section] = {}
            current_subsec = None
            current_list = None
        elif indent == 2 and current_section is not None:
            if stripped.endswith(":"):
                current_subsec = stripped[:-1].strip()
                result[current_section][current_subsec] = {}
                current_list = None
            elif ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                # 类型推断
                if v.isdigit():
                    val: Any = int(v)
                else:
                    try:
                        val = float(v)
                    except ValueError:
                        if v.lower() == "true":
                            val = True
                        elif v.lower() == "false":
                            val = False
                        else:
                            val = v
                result[current_section][k] = val
                current_subsec = None
                current_list = None
        elif indent == 4 and current_section is not None:
            if stripped.startswith("- "):
                item = stripped[2:].strip().strip("\"'")
                if current_subsec is not None:
                    if not isinstance(result[current_section].get(current_subsec), list):
                        result[current_section][current_subsec] = []
                    result[current_section][current_subsec].append(item)
            elif ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip().strip("\"'")
                v = v.strip().strip("\"'")
                if current_subsec is not None:
                    if not isinstance(result[current_section].get(current_subsec), dict):
                        result[current_section][current_subsec] = {}
                    result[current_section][current_subsec][k] = v
    return result


@dataclass
class RuntimeConfig:
    venv_path: str = field(
        default_factory=lambda: str(Path.home() / ".local" / "share" / "interview-analyzer" / ".venv")
    )
    python: str = "python3"
    ffmpeg: str = "/opt/homebrew/bin/ffmpeg"
    ffprobe: str = "/opt/homebrew/bin/ffprobe"


@dataclass
class NetworkConfig:
    pypi_mirror: str = "https://pypi.tuna.tsinghua.edu.cn/simple"
    hf_mirror: str = "https://hf-mirror.com"


@dataclass
class InterviewConfig:
    domain: str = "frontend"
    vocab_prompt: str = (
        "AI Coding, Claude Code, Agent, Planner, Reviewer, Hooks, Skills, "
        "Subagent, React, SSE, RAG, Node.js, TypeScript, Webpack, Vite"
    )
    asr_term_map: Dict[str, str] = field(
        default_factory=lambda: {
            "reg": "RAG",
            "Cloud Code": "Claude Code",
            "上约文": "上下文",
        }
    )


@dataclass
class ThresholdsConfig:
    disk_min_gb: float = 5.0
    coverage_auto: float = 0.4
    loop_repeat_min: int = 3
    loop_text_min_chars: int = 8
    silence_energy_window_ms: int = 100
    silence_rms_threshold: float = 0.005
    timestamp_gap_seconds: float = 30.0


@dataclass
class IndexingConfig:
    ai_coding_keywords: List[str] = field(
        default_factory=lambda: [
            "AI", "Coding", "Agent", "Planner", "Reviewer", "Hook", "Hooks",
            "Skill", "Skills", "Subagent", "工作流", "prompt", "Prompt",
            "MCP", "Claude Code"
        ]
    )
    project_keywords: List[str] = field(
        default_factory=lambda: [
            "项目", "负责", "主导", "难点", "指标", "收益", "架构", "方案",
            "上线", "迭代", "重构", "优化", "团队", "需求"
        ]
    )
    question_patterns: List[str] = field(
        default_factory=lambda: [
            "为什么", "怎么", "如何", "有没有", "介绍一下", "具体说说"
        ]
    )
    reverse_question_patterns: List[str] = field(
        default_factory=lambda: [
            "还有什么问题", "有什么想问", "有什么想了解", "想问一下", "想了解一下",
            "了解一下", "待遇", "薪资", "加班", "福利", "晋升", "转正",
            "入职", "几轮面试", "团队规模", "业务方向", "技术栈"
        ]
    )


@dataclass
class ModelsConfig:
    mlx_whisper: str = "0.4.3"
    small: str = "mlx-community/whisper-small-mlx"
    small_revision: str = "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e"
    turbo: str = "mlx-community/whisper-large-v3-turbo"
    turbo_revision: str = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"


@dataclass
class AnalyzerConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    interview: InterviewConfig = field(default_factory=InterviewConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    config_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "runtime": {
                "venv_path": self.runtime.venv_path,
                "python": self.runtime.python,
                "ffmpeg": self.runtime.ffmpeg,
                "ffprobe": self.runtime.ffprobe,
            },
            "network": {
                "pypi_mirror": self.network.pypi_mirror,
                "hf_mirror": self.network.hf_mirror,
            },
            "interview": {
                "domain": self.interview.domain,
                "vocab_prompt": self.interview.vocab_prompt,
                "asr_term_map": self.interview.asr_term_map,
            },
            "thresholds": {
                "disk_min_gb": self.thresholds.disk_min_gb,
                "coverage_auto": self.thresholds.coverage_auto,
                "loop_repeat_min": self.thresholds.loop_repeat_min,
                "loop_text_min_chars": self.thresholds.loop_text_min_chars,
                "silence_energy_window_ms": self.thresholds.silence_energy_window_ms,
                "silence_rms_threshold": self.thresholds.silence_rms_threshold,
                "timestamp_gap_seconds": self.thresholds.timestamp_gap_seconds,
            },
            "indexing": {
                "ai_coding_keywords": self.indexing.ai_coding_keywords,
                "project_keywords": self.indexing.project_keywords,
                "question_patterns": self.indexing.question_patterns,
                "reverse_question_patterns": self.indexing.reverse_question_patterns,
            },
            "models": {
                "mlx_whisper": self.models.mlx_whisper,
                "small": self.models.small,
                "small_revision": self.models.small_revision,
                "turbo": self.models.turbo,
                "turbo_revision": self.models.turbo_revision,
            },
        }


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归深度合并两个配置字典，override 优先级高于 base"""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def find_default_config_path() -> Optional[Path]:
    """寻找默认配置文件路径"""
    candidates = [
        Path.cwd() / "config" / "analyzer-config.yaml",
        Path(__file__).resolve().parent.parent.parent / "config" / "analyzer-config.yaml",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def load_config(config_path: Optional[str | Path] = None) -> AnalyzerConfig:
    """加载配置并返回 AnalyzerConfig 实例（支持自动发现并深度合并 analyzer-config.local.yaml）"""
    resolved_path: Optional[Path] = None
    if config_path:
        p = Path(config_path).expanduser().resolve()
        if p.exists() and p.is_file():
            resolved_path = p
        else:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
    else:
        resolved_path = find_default_config_path()

    if not resolved_path:
        # 使用默认配置
        return AnalyzerConfig()

    with open(resolved_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    if yaml is not None:
        data = yaml.safe_load(file_content) or {}
    else:
        data = _simple_yaml_parse(file_content)

    # 自动探测并深度合并本地私有配置文件 analyzer-config.local.yaml
    local_candidates = [
        resolved_path.parent / "analyzer-config.local.yaml",
        Path.cwd() / "config" / "analyzer-config.local.yaml",
        Path(__file__).resolve().parent.parent.parent / "config" / "analyzer-config.local.yaml",
    ]
    for local_p in local_candidates:
        if local_p.exists() and local_p.is_file():
            with open(local_p, "r", encoding="utf-8") as f_local:
                local_content = f_local.read()
            if yaml is not None:
                local_data = yaml.safe_load(local_content) or {}
            else:
                local_data = _simple_yaml_parse(local_content)
            data = _deep_merge_dicts(data, local_data)
            break

    runtime_data = data.get("runtime", {})
    network_data = data.get("network", {})
    interview_data = data.get("interview", {})
    thresholds_data = data.get("thresholds", {})
    indexing_data = data.get("indexing", {})
    models_data = data.get("models", {})

    # 自动推导 ffprobe 路径（如果仅配了 ffmpeg）
    ffmpeg_bin = runtime_data.get("ffmpeg", "/opt/homebrew/bin/ffmpeg")
    ffprobe_bin = runtime_data.get("ffprobe")
    if not ffprobe_bin:
        ffmpeg_parent = Path(ffmpeg_bin).parent
        candidate_ffprobe = ffmpeg_parent / "ffprobe"
        ffprobe_bin = str(candidate_ffprobe) if candidate_ffprobe.exists() else "ffprobe"

    raw_venv = runtime_data.get("venv_path", "~/.local/share/interview-analyzer/.venv")
    expanded_venv = str(Path(raw_venv).expanduser())

    runtime = RuntimeConfig(
        venv_path=expanded_venv,
        python=runtime_data.get("python", "python3"),
        ffmpeg=ffmpeg_bin,
        ffprobe=ffprobe_bin,
    )

    network = NetworkConfig(
        pypi_mirror=network_data.get("pypi_mirror", "https://pypi.tuna.tsinghua.edu.cn/simple"),
        hf_mirror=network_data.get("hf_mirror", "https://hf-mirror.com"),
    )

    interview = InterviewConfig(
        domain=interview_data.get("domain", "frontend"),
        vocab_prompt=interview_data.get(
            "vocab_prompt",
            "AI Coding, Claude Code, Agent, Planner, Reviewer, Hooks, Skills, "
            "Subagent, React, SSE, RAG, Node.js, TypeScript, Webpack, Vite"
        ),
        asr_term_map=interview_data.get(
            "asr_term_map",
            {"reg": "RAG", "Cloud Code": "Claude Code", "上约文": "上下文"}
        ),
    )

    thresholds = ThresholdsConfig(
        disk_min_gb=float(thresholds_data.get("disk_min_gb", 5.0)),
        coverage_auto=float(thresholds_data.get("coverage_auto", 0.4)),
        loop_repeat_min=int(thresholds_data.get("loop_repeat_min", 3)),
        loop_text_min_chars=int(thresholds_data.get("loop_text_min_chars", 8)),
        silence_energy_window_ms=int(thresholds_data.get("silence_energy_window_ms", 100)),
        silence_rms_threshold=float(thresholds_data.get("silence_rms_threshold", 0.005)),
        timestamp_gap_seconds=float(thresholds_data.get("timestamp_gap_seconds", 30.0)),
    )

    indexing = IndexingConfig(
        ai_coding_keywords=indexing_data.get("ai_coding_keywords", [
            "AI", "Coding", "Agent", "Planner", "Reviewer", "Hook", "Hooks",
            "Skill", "Skills", "Subagent", "工作流", "prompt", "Prompt",
            "MCP", "Claude Code"
        ]),
        project_keywords=indexing_data.get("project_keywords", [
            "项目", "负责", "主导", "难点", "指标", "收益", "架构", "方案",
            "上线", "迭代", "重构", "优化", "团队", "需求"
        ]),
        question_patterns=indexing_data.get("question_patterns", [
            "为什么", "怎么", "如何", "有没有", "介绍一下", "具体说说"
        ]),
        reverse_question_patterns=indexing_data.get("reverse_question_patterns", [
            "还有什么问题", "有什么想问", "有什么想了解", "想问一下", "想了解一下",
            "了解一下", "待遇", "薪资", "加班", "福利", "晋升", "转正",
            "入职", "几轮面试", "团队规模", "业务方向", "技术栈"
        ]),
    )

    models = ModelsConfig(
        mlx_whisper=str(models_data.get("mlx_whisper", "0.4.3")),
        small=str(models_data.get("small", "mlx-community/whisper-small-mlx")),
        small_revision=str(models_data.get(
            "small_revision",
            "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e"
        )),
        turbo=str(models_data.get("turbo", "mlx-community/whisper-large-v3-turbo")),
        turbo_revision=str(models_data.get(
            "turbo_revision",
            "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
        )),
    )

    return AnalyzerConfig(
        runtime=runtime,
        network=network,
        interview=interview,
        thresholds=thresholds,
        indexing=indexing,
        models=models,
        config_file_path=str(resolved_path),
    )
