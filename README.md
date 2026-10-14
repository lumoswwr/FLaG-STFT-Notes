# FLaG-STFT-Notes

38 项已完成实验的 MkDocs Material 重构版。

- 本版每项实验独立 Markdown，起始部分原样保留提交的实验总结。
- 实验页支持“总结 → 详细论述 → 返回总结”的双向导航。
- 原文档完整归档在仓库根目录 `archive/legacy/`。
- 实验代码核对仓库：[AMPCliff / FLaG-STFT-mechanism](https://github.com/lumoswwr/AMPCliff/tree/FLaG-STFT-mechanism)。

## 本地浏览

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdocs serve
```

浏览器打开命令行显示的本地地址。使用 `mkdocs build --strict` 检查站点构建。
