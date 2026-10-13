# FLaG / STFT Mechanism Experiments

这是本项目实验笔记的 Markdown 源仓库，使用 MkDocs Material 发布。网页：[FLaG / STFT 实验报告](https://lumoswwr.github.io/FLaG-STFT-Notes/)。

## 怎么读报告？

网站首页先用几段普通语言解释研究脉络；七篇实验报告的开头都列出了**实验目的、思路、结果和结论**，共 38 项。需要复查原始配置、公式和数据时，再阅读每篇下面的“详细实验记录与数据”。

主要路径：

- `docs/index.md`：研究首页及七篇实验报告的阅读顺序。
- `docs/findings.md`：阶段性结果总结及关键数值。
- `docs/methods.md`：统一实验设置和不同阶段的配置差异。
- `docs/stsb/`：STSB 上的 STFT 结构、Padding 与 Global/Local 机制分析。
- `docs/sprint/`：Sprint 的训练协议与 Global/Local 性能对照。
- `docs/dc-frequency.md`：DC 和频带对预测的作用。
- `docs/mean-compatible-flag.md`：Mean 与 FLaG 的融合方法。
- `docs/flag-gate-projection.md`：Gate 和可训练线性层的作用分析。

## 在本地运行（macOS / Linux）

```bash
cd FLaG-STFT-Notes
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdocs serve
```

在终端提示的地址（通常是 http://127.0.0.1:8000）查看本地网站。

生成静态 HTML：

```bash
mkdocs build
```

推送到 `main` 后，会由仓库 GitHub Actions 的工作流自动尝试部署 GitHub Pages。
