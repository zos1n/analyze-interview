"""
Pytest Fixtures 共享配置与测试数据集
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from analyze_interview.config import AnalyzerConfig, load_config


@pytest.fixture
def sample_config() -> AnalyzerConfig:
    """提供标准默认配置"""
    return load_config()


@pytest.fixture
def synthetic_pcm_16k() -> np.ndarray:
    """生成 10 秒测试用的合成 PCM 数据 (前 3 秒静音, 中间 4 秒正弦波语音, 后 3 秒静音)"""
    sample_rate = 16000
    duration = 10
    total_samples = sample_rate * duration

    pcm = np.zeros(total_samples, dtype=np.float32)

    # 3.0s 到 7.0s 注入有效音频信号 (440Hz 正弦波，振幅 0.2)
    t = np.linspace(0, 4, sample_rate * 4, endpoint=False, dtype=np.float32)
    speech_wave = 0.2 * np.sin(2 * np.pi * 440 * t)
    pcm[3 * sample_rate : 7 * sample_rate] = speech_wave

    return pcm


@pytest.fixture
def sample_segments() -> List[Dict[str, Any]]:
    """提供用于测试的模拟转写段落数据"""
    return [
        {"id": 0, "start": 0.0, "end": 2.5, "text": "字幕制作：点点栏目"},
        {"id": 1, "start": 3.0, "end": 8.0, "text": "你能先简单介绍一下你自己吗？"},
        {"id": 2, "start": 8.5, "end": 25.0, "text": "好的，我叫张三，有5年前端及AI应用开发经验，主要负责大模型应用与Agent架构。"},
        {"id": 3, "start": 26.0, "end": 35.0, "text": "那你们在项目中SSE流式通信和打字机渲染是怎么做性能优化的？"},
        {"id": 4, "start": 36.0, "end": 50.0, "text": "我们在前端应用中设计了调度缓冲区与分帧打字机渲染机制。"},
        {"id": 5, "start": 55.0, "end": 58.0, "text": "嗯"},
        {"id": 6, "start": 58.0, "end": 61.0, "text": "嗯"},
        {"id": 7, "start": 61.0, "end": 64.0, "text": "嗯"},
        {"id": 8, "start": 64.0, "end": 67.0, "text": "嗯"},
        {"id": 9, "start": 100.0, "end": 110.0, "text": "你还有什么想了解的业务方向或者团队技术栈吗？"},
        {"id": 10, "start": 112.0, "end": 120.0, "text": "我想了解一下团队目前AI Coding和前端工程化在团队内部的落地情况。"},
    ]
