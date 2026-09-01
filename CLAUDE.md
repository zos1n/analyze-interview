# CLAUDE.md

This file provides guidance to AI coding assistants when working with code in this repository.

## 项目定位

面试录屏分析工具 (`analyze-interview`) —— 基于 Apple Silicon MLX 硬件加速的**全本地、零云端费用、隐私保护**的面试录屏音轨提取、ASR 语音转写、病理修复、问题索引、Turbo 局部精转与逐题结构化复盘系统。

## 规范与执行依据

- 完整规范：`docs/面试录屏分析工具规范.md` (v2.1)
- 共享配置：`config/analyzer-config.yaml`
- 规则底线：`AGENTS.md`
- 历史方案与归档：`docs/archive/`（临时方案/任务文件落地并验证通过后，自动移动归档至此）
- 历史存档：`docs/面试录屏分析执行方案-历史存档.md`

## 技术栈与设计原则

- **语言环境**：Python 3.9+ (兼容 macOS 自带 3.9.6 与 Apple Silicon arm64)
- **底层加速**：FFmpeg / FFprobe + Apple Silicon MLX (`mlx-whisper==0.4.3`)
- **零重型外部依赖**：仅使用 `numpy` 进行 PCM RMS 能量计算与相关系数分析，不引入 `soundfile` / `librosa` 等多余底层 C 库。
- **不可违背底线**：
  1. 绝不覆盖、修改或删除原视频文件，开工前与收工后均复核 SHA256。
  2. 绝不使用云端 ASR / 视频大模型，绝不把完整转写直传分析模型。
  3. 绝不删除中间产物，支持随时断点恢复 (`execution-state.json`)。
  4. 多区间转录严禁拼接单个 `--clip-timestamps`（避开 mlx-whisper 0.4.3 multi-clip seek 连续转录 bug）。
  5. Turbo 精转默认加 `--condition-on-previous-text False`。

## 常用开发命令

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装开发与测试依赖
pip install -r requirements-dev.txt

# 运行全部单元测试
pytest tests/ -v

# 运行代码风格检查
ruff check src/ tests/

# CLI 执行方式
python -m analyze_interview --help
python -m analyze_interview run --video /path/to/interview.mp4
```
