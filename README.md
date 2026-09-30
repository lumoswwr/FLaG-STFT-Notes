# FLaG-STFT-Notes

由现有 Word 实验报告迁移得到的 MkDocs Material 项目。

## 第一次运行（macOS）

```bash
cd FLaG-STFT-Notes
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdocs serve
```

浏览器打开终端显示的地址，通常是：

```text
http://127.0.0.1:8000
```

以后重新打开项目：

```bash
cd FLaG-STFT-Notes
source .venv/bin/activate
mkdocs serve
```

生成静态 HTML：

```bash
mkdocs build
```
