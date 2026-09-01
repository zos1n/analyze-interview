"""Stage 0 预检与基线计算单元测试"""

import hashlib
from pathlib import Path

from analyze_interview.config import AnalyzerConfig
from analyze_interview.precheck import (
    calculate_video_baseline,
    check_disk_space,
    derive_network_decisions,
)


def test_calculate_video_baseline(tmp_path: Path):
    dummy_video = tmp_path / "test_video.mp4"
    content = b"fake video content for sha256 testing"
    dummy_video.write_bytes(content)

    base = calculate_video_baseline(dummy_video)
    expected_sha = hashlib.sha256(content).hexdigest()

    assert base["sha256"] == expected_sha
    assert base["size_bytes"] == len(content)
    assert base["video_path"] == str(dummy_video.resolve())


def test_check_disk_space(tmp_path: Path):
    disk = check_disk_space(tmp_path, min_gb=0.01)
    assert disk["passed"] is True
    assert disk["free_gb"] > 0.0


def test_derive_network_decisions(sample_config: AnalyzerConfig):
    # 测试直连均通的情况
    net_matrix = {
        "pypi_org": {"curl": {"success": True}, "python": {"success": True}},
        "huggingface_co": {"curl": {"success": True}, "python": {"success": True}},
    }
    proxy_info = {"configured": False, "reachable": False}
    decisions = derive_network_decisions(net_matrix, proxy_info, sample_config)

    assert decisions["pip_index_url"] == "https://pypi.org/simple"
    assert decisions["hf_endpoint"] is None
    assert "NO_PROXY" not in decisions["env_vars"]

    # 测试失效代理与 HF 超时
    net_matrix_fail = {
        "pypi_org": {"curl": {"success": True}, "python": {"success": False}},
        "huggingface_co": {"curl": {"success": False}, "python": {"success": False}},
    }
    proxy_info_dead = {"configured": True, "reachable": False, "host": "127.0.0.1", "port": 9999}
    decisions_fail = derive_network_decisions(net_matrix_fail, proxy_info_dead, sample_config)

    assert decisions_fail["env_vars"].get("NO_PROXY") == "*"
    assert decisions_fail["hf_endpoint"] == sample_config.network.hf_mirror
