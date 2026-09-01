# 面试录屏分析工具 (analyze-interview)

> 基于 Apple Silicon MLX 硬件加速与 Whisper 的本地化面试录屏智能复盘系统。

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-orange.svg)]()

## 🌟 核心特性

- **🔒 100% 本地运行 & 零云端费用**：无需任何云端 ASR 或视频大模型 API，所有音频抽取、Whisper ASR 转写及病理修复均在本地 Apple Silicon（MLX）上完成，保护求职者个人隐私。
- **🔐 双层配置隔离与私有映射（Local Override）**：公共默认配置（`analyzer-config.yaml`）保持纯净通用；本地私有配置（`analyzer-config.local.yaml`，已写入 `.gitignore`）可在运行时自动深度合并注入专属人名、目标公司与 ASR 纠偏词表，保障开源安全与本地精准度的完美统一。
- **🎯 智能多音轨识别与能量互补分析**：逐秒（1s）RMS 能量剖面分析与轨间互补度计算，自动识别系统音频（面试官）与麦克风（候选人）分轨录制，避免单轨静音误判为未作答。
- **🛠️ 转写病理检测与自愈修复**：
  - 基于裸 PCM 的均方根 (RMS) 静音文本幻觉过滤。
  - 重复循环检测（叠加 RMS 区分口吃与幻觉），结合 `--condition-on-previous-text False` 实现窄区间定向重转。
  - 确定性片段合并算法（修复段整段保留不截断、修复段优先、时间戳自愈排序）。
- **📊 问句索引与密度聚类 (40% 覆盖率自动决策)**：
  - ASCII 词边界与中文子串关键词聚类，避免“AI”等词命中英文子串。
  - 扩展中文语法问句检测（句末助词、疑问代词、反问追问引导）与回答排除规则。
  - 自动计算回答窗口（`Q_i.end → Q_{i+1}.start`）及必须包含区间（开头、反问环节、结尾）。
- **⚡ 规避 Whisper 已知踩坑**：
  - 针对 `mlx-whisper 0.4.3` 的多区间 seek 连续转录 bug，自动隔离为单命令单区间处理并智能合并。
  - 修复文件名整数边界命名，避免 `.NN` 小数后缀被截断覆盖。
- **📝 结构化逐题复盘**：
  - 自动提炼 `transcript-key.md`，对未作答区间进行 0.5s RMS 扫描与语气分类重转。
  - 支持 `codex` / `api` / `self` 三种模式，输出 7 维度复盘分析（`interview-review.md`），严格遵守事实/推断/无法判断三分法。
- **🛡️ 严格的完整性校验与断点续跑**：开工与验收比对视频 SHA-256，`execution-state.json` 支持全流程任意断点恢复。

---

## 📂 项目结构

```text
analyze-interview/
├── config/
│   └── analyzer-config.yaml    # 默认共享配置文件
├── docs/
│   ├── 面试录屏分析工具规范.md  # 唯一执行规范 (v2.1)
│   ├── 面试录屏分析执行方案-历史存档.md
│   └── AGENTS-template.md
├── commands/
│   └── analyze-interview.md    # 斜杠命令入口定义
├── src/
│   └── analyze_interview/
│       ├── __init__.py
│       ├── config.py           # 配置读取与数据类
│       ├── state.py            # 状态机与断点恢复
│       ├── precheck.py         # 预检矩阵、代理检测、SHA256 基线
│       ├── environment.py      # 环境依赖与模型缓存核验
│       ├── audio.py            # 音轨提取、能量剖面、分轨/声道判定
│       ├── transcribe.py       # Whisper Small/Turbo MLX 驱动
│       ├── pathology.py        # RMS 静音检测、循环检测、确定性合并
│       ├── indexer.py          # 问题索引、密度聚类、回答窗口
│       ├── key_transcript.py   # 关键问答稿生成、说话人归属
│       ├── reviewer.py         # 逐题分析与提示词构建
│       ├── verifier.py         # 验收校验器
│       ├── pipeline.py         # 流水线总控编排
│       └── cli.py              # CLI 命令行入口
├── tests/                      # 完整自动化测试套件
├── pyproject.toml              # 项目打包配置
├── requirements.txt            # 依赖清单
├── README.md
├── CLAUDE.md
└── AGENTS.md
```

---

## 🚀 快速上手

### 1. 环境准备

推荐在 macOS (Apple Silicon M系列芯片) 下运行：

```bash
# 安装系统依赖
brew install ffmpeg

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 命令行使用

```bash
# 查看帮助
python -m analyze_interview --help

# 运行完整分析流水线
python -m analyze_interview run --video /path/to/interview.mp4

# 单步运行（如仅做预检或音轨提取）
python -m analyze_interview precheck --video /path/to/interview.mp4
python -m analyze_interview extract-audio --video /path/to/interview.mp4

# 从中断点继续恢复执行
python -m analyze_interview run --video /path/to/interview.mp4 --resume
```

### 3. 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 📄 许可协议

MIT License.
