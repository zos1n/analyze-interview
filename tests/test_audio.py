"""音频与多轨/声道分析单元测试"""

import numpy as np

from analyze_interview.audio import (
    analyze_multitrack_complementarity,
)


def test_multitrack_complementarity_split_mode():
    # 模拟 100 秒的多轨数据：
    # Track 0 (面试官): 0-30s 有声, 30-70s 静音, 70-100s 有声
    # Track 1 (候选人): 0-30s 静音, 30-70s 有声, 70-100s 静音
    p0 = np.zeros(100, dtype=np.float32)
    p0[0:30] = 0.05
    p0[70:100] = 0.05

    p1 = np.zeros(100, dtype=np.float32)
    p1[30:70] = 0.08

    res = analyze_multitrack_complementarity([p0, p1], silence_threshold=0.005)
    assert res["is_split_tracks"] is True
    assert res["mode"] == "split_tracks"
    assert res["complementary_ratio"] == 1.0
    assert res["correlation"] < 0.0


def test_multitrack_redundant_mode():
    # 模拟两条高度相同的音轨
    p0 = np.array([0.05, 0.08, 0.0, 0.0, 0.02] * 20, dtype=np.float32)
    p1 = p0 * 0.95  # 几乎完全一样

    res = analyze_multitrack_complementarity([p0, p1], silence_threshold=0.005)
    assert res["is_split_tracks"] is False
    assert res["mode"] == "redundant_tracks"
    assert res["correlation"] > 0.95
