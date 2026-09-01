"""
Stage 0 预检模块：负责原视频 SHA256 基线、磁盘空间检查、网络双测矩阵与代理自愈决策。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

from analyze_interview.config import AnalyzerConfig


def calculate_video_baseline(video_path: str | Path) -> Dict[str, Any]:
    """计算原视频的基线信息 (size + mtime + SHA256)"""
    v_path = Path(video_path).expanduser().resolve()
    if not v_path.exists() or not v_path.is_file():
        raise FileNotFoundError(f"原视频文件不存在: {video_path}")

    stat = v_path.stat()
    size_bytes = stat.st_size
    mtime = stat.st_mtime

    # 流式计算 SHA256
    sha256_hash = hashlib.sha256()
    with open(v_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    sha256_hex = sha256_hash.hexdigest()

    return {
        "video_path": str(v_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "mtime": mtime,
        "sha256": sha256_hex,
    }


def check_disk_space(target_dir: str | Path, min_gb: float = 5.0) -> Dict[str, Any]:
    """检查目标目录所在挂载点的剩余磁盘空间"""
    t_dir = Path(target_dir).expanduser().resolve()
    # 如果目录尚不存在，取上层已存在的目录
    curr = t_dir
    while not curr.exists() and curr != curr.parent:
        curr = curr.parent

    total, used, free = shutil.disk_usage(curr)
    free_gb = free / (1024 ** 3)
    passed = free_gb >= min_gb
    return {
        "target_dir": str(t_dir),
        "free_gb": round(free_gb, 2),
        "min_required_gb": min_gb,
        "passed": passed,
    }


def _test_curl(url: str, timeout: int = 5) -> Dict[str, Any]:
    """使用系统 curl 命令测试 HTTP 连通性"""
    start_t = time.time()
    try:
        res = subprocess.run(
            [
                "curl", "-s", "-o", "/dev/null",
                "-w", "%{http_code}",
                "--connect-timeout", str(timeout),
                "-m", str(timeout),
                url
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        elapsed_ms = round((time.time() - start_t) * 1000, 1)
        code = int(res.stdout.strip()) if res.stdout.strip().isdigit() else 0
        return {
            "method": "curl",
            "success": 200 <= code < 400,
            "status_code": code,
            "latency_ms": elapsed_ms,
            "error": None if (200 <= code < 400) else f"HTTP code {code}",
        }
    except Exception as e:
        elapsed_ms = round((time.time() - start_t) * 1000, 1)
        return {
            "method": "curl",
            "success": False,
            "status_code": 0,
            "latency_ms": elapsed_ms,
            "error": str(e),
        }


def _test_python_urllib(url: str, timeout: int = 5, env_no_proxy: bool = False) -> Dict[str, Any]:
    """使用 Python 内置 urllib 测试 HTTP 连通性（读取环境变量代理）"""
    start_t = time.time()
    old_no_proxy = os.environ.get("NO_PROXY")
    try:
        if env_no_proxy:
            os.environ["NO_PROXY"] = "*"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "analyze-interview-precheck/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            code = response.getcode()
            elapsed_ms = round((time.time() - start_t) * 1000, 1)
            return {
                "method": "python_urllib",
                "success": 200 <= code < 400,
                "status_code": code,
                "latency_ms": elapsed_ms,
                "error": None,
            }
    except Exception as e:
        elapsed_ms = round((time.time() - start_t) * 1000, 1)
        return {
            "method": "python_urllib",
            "success": False,
            "status_code": 0,
            "latency_ms": elapsed_ms,
            "error": str(e),
        }
    finally:
        if env_no_proxy:
            if old_no_proxy is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = old_no_proxy


def check_proxy_status() -> Dict[str, Any]:
    """检测 macOS 系统代理与 Python 代理生效状态，并诊断代理可用性"""
    sys_proxy_info: Dict[str, Any] = {"configured": False, "reachable": False, "details": {}}

    # 1. 读 macOS scutil --proxy
    try:
        res = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            lines = res.stdout.splitlines()
            parsed = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    parsed[k.strip()] = v.strip()
            sys_proxy_info["details"]["scutil"] = parsed

            # 检查是否有激活的 HTTP / HTTPS 代理
            http_enabled = parsed.get("HTTPEnable") == "1"
            https_enabled = parsed.get("HTTPSEnable") == "1"
            proxy_host = parsed.get("HTTPProxy") or parsed.get("HTTPSProxy")
            proxy_port = parsed.get("HTTPPort") or parsed.get("HTTPSPort")

            if (http_enabled or https_enabled) and proxy_host and proxy_port:
                sys_proxy_info["configured"] = True
                sys_proxy_info["host"] = proxy_host
                sys_proxy_info["port"] = int(proxy_port) if str(proxy_port).isdigit() else 0

                # 探测代理端口连通性
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1.0)
                    s.connect((proxy_host, int(proxy_port)))
                    s.close()
                    sys_proxy_info["reachable"] = True
                except Exception:
                    sys_proxy_info["reachable"] = False
    except Exception as e:
        sys_proxy_info["details"]["scutil_error"] = str(e)

    # 2. 读 Python urllib getproxies
    py_proxies = urllib.request.getproxies()
    sys_proxy_info["details"]["python_proxies"] = py_proxies

    return sys_proxy_info


def check_network_matrix(config: AnalyzerConfig, timeout: int = 8) -> Dict[str, Any]:
    """执行规范要求的四目标网络连通性矩阵双测"""
    targets = {
        "pypi_org": "https://pypi.org/simple/",
        "pypi_tuna": config.network.pypi_mirror,
        "huggingface_co": "https://huggingface.co",
        "hf_mirror": config.network.hf_mirror,
    }

    matrix: Dict[str, Any] = {}
    for name, url in targets.items():
        curl_res = _test_curl(url, timeout=timeout)
        py_res = _test_python_urllib(url, timeout=timeout)
        matrix[name] = {
            "url": url,
            "curl": curl_res,
            "python": py_res,
        }
    return matrix


def derive_network_decisions(
    net_matrix: Dict[str, Any],
    proxy_info: Dict[str, Any],
    config: AnalyzerConfig,
) -> Dict[str, Any]:
    """根据规范自动决策表推导网络环境变量与镜像策略"""
    env_vars: Dict[str, str] = {}
    notes: list[str] = []

    # 1. 代理判定
    if proxy_info.get("configured"):
        if not proxy_info.get("reachable"):
            env_vars["NO_PROXY"] = "*"
            notes.append("系统代理已配置但端口不可达（失效代理），全流程注入 NO_PROXY='*'")
        else:
            notes.append(f"系统代理正常运作 ({proxy_info.get('host')}:{proxy_info.get('port')})")

    # 2. PyPI 源决策
    pypi_org = net_matrix.get("pypi_org", {})
    pypi_org_curl = pypi_org.get("curl", {}).get("success", False)
    pypi_org_py = pypi_org.get("python", {}).get("success", False)

    if pypi_org_curl and pypi_org_py:
        pip_index_url = "https://pypi.org/simple"
        notes.append("PyPI 直连连通良好")
    elif pypi_org_curl and not pypi_org_py:
        # curl 通但 Python 失败：说明 Python 侧受代理或环境干扰
        # 尝试 NO_PROXY 补测
        py_no_proxy_res = _test_python_urllib("https://pypi.org/simple/", timeout=5, env_no_proxy=True)
        if py_no_proxy_res.get("success"):
            env_vars["NO_PROXY"] = "*"
            pip_index_url = "https://pypi.org/simple"
            notes.append("Python 受失效代理干扰，注入 NO_PROXY='*' 后 PyPI 直连恢复")
        else:
            pip_index_url = config.network.pypi_mirror
            notes.append(f"PyPI 直连不稳定，切换至清华镜像: {pip_index_url}")
    else:
        pip_index_url = config.network.pypi_mirror
        notes.append(f"PyPI 官方源不可达，切换至清华镜像: {pip_index_url}")

    # 3. Hugging Face 源决策
    hf_co = net_matrix.get("huggingface_co", {})
    hf_co_ok = hf_co.get("curl", {}).get("success", False) or hf_co.get("python", {}).get("success", False)

    if hf_co_ok:
        hf_endpoint = None
        notes.append("Hugging Face 官方源直连可达")
    else:
        hf_endpoint = config.network.hf_mirror
        env_vars["HF_ENDPOINT"] = hf_endpoint
        notes.append(f"Hugging Face 官方源超时/不可达，启用镜像: {hf_endpoint}")

    return {
        "env_vars": env_vars,
        "pip_index_url": pip_index_url,
        "hf_endpoint": hf_endpoint,
        "notes": notes,
    }


def run_precheck(
    video_path: str | Path,
    output_dir: str | Path,
    config: AnalyzerConfig,
) -> Dict[str, Any]:
    """完整运行 Stage 0 预检流程"""
    # 1. 计算原视频基线
    baseline = calculate_video_baseline(video_path)

    # 2. 检查产物目录所在磁盘空间
    disk = check_disk_space(output_dir, min_gb=config.thresholds.disk_min_gb)
    if not disk["passed"]:
        raise RuntimeError(
            f"磁盘空间不足: 剩余 {disk['free_gb']}GB, 规范要求至少 {disk['min_required_gb']}GB"
        )

    # 3. 检测代理状态
    proxy_info = check_proxy_status()

    # 4. 检测网络矩阵
    net_matrix = check_network_matrix(config)

    # 5. 自动推导决策
    decisions = derive_network_decisions(net_matrix, proxy_info, config)

    return {
        "baseline": baseline,
        "disk": disk,
        "proxy_info": proxy_info,
        "network_matrix": net_matrix,
        "decisions": decisions,
    }
