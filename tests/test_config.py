"""配置模块单元测试"""

import pytest

from analyze_interview.config import AnalyzerConfig, load_config


def test_load_default_config(sample_config: AnalyzerConfig):
    assert sample_config is not None
    assert sample_config.runtime.python == "python3"
    assert sample_config.models.mlx_whisper == "0.4.3"
    assert sample_config.thresholds.coverage_auto == 0.4
    assert sample_config.thresholds.silence_rms_threshold == 0.005
    assert "AI" in sample_config.indexing.ai_coding_keywords
    assert "项目" in sample_config.indexing.project_keywords


def test_config_to_dict(sample_config: AnalyzerConfig):
    d = sample_config.to_dict()
    assert isinstance(d, dict)
    assert "runtime" in d
    assert "thresholds" in d
    assert "indexing" in d
    assert d["thresholds"]["loop_repeat_min"] == 3


def test_load_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_config("/path/to/non_existent_config_123.yaml")


def test_load_config_with_local_override(tmp_path):
    base_config_path = tmp_path / "analyzer-config.yaml"
    base_config_path.write_text("""
runtime:
  python: python3
  venv_path: ~/.local/share/interview-analyzer/.venv
interview:
  domain: frontend
  vocab_prompt: "React, Vue"
  asr_term_map:
    reg: RAG
thresholds:
  disk_min_gb: 5
""", encoding="utf-8")

    local_config_path = tmp_path / "analyzer-config.local.yaml"
    local_config_path.write_text("""
interview:
  vocab_prompt: "CustomPrompt, React, Vue"
  asr_term_map:
    自定义错词: 自定义正词
""", encoding="utf-8")

    cfg = load_config(base_config_path)
    # 验证深度合并生效：
    # 1. 覆盖字段生效
    assert cfg.interview.vocab_prompt == "CustomPrompt, React, Vue"
    assert cfg.interview.asr_term_map.get("自定义错词") == "自定义正词"
    # 2. 基础配置中的旧字段未被覆盖冲掉
    assert cfg.interview.asr_term_map.get("reg") == "RAG"
    assert cfg.runtime.python == "python3"
    assert cfg.thresholds.disk_min_gb == 5.0

