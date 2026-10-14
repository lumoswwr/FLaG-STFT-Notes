# FLaG-STFT-Notes

由 MkDocs Material 构建的实验记录。**重构后的实验主导航按每个实验单独成页组织**，每页为作者四项总结预留固定位置（总结原文暂未提供），随后通过页内链接进入详细论述。

- [研究首页](docs/index.md)
- [逐项实验索引](docs/experiments/index.md)
- [结果索引](docs/findings.md)
- [协议与指标](docs/methods.md)
- [实验代码仓库](https://github.com/lumoswwr/AMPCliff/tree/FLaG-STFT-mechanism)

## 维护规则

1. 将作者提供的“实验目的/实验思路/实验结果/实验结论”**逐字**粘贴至各实验 Markdown 文件顶部四字段，不重写文字。
2. 页内详细论述单列实验设置、数据、结果与能够支持的判断。更新数值时在代码仓库中核对对应 seed、split 与 checkpoint 规则。
3. 旧版实验记录仍保留原路径用于历史溯源，但不在新导航中展示，以减少重复和混合协议。
4. 原有研究者个人思考笔记 `docs/MTEB.md` 未并入事实结论，仍保存在仓库中。

## 本地查看

```bash
python3 -m pip install -r requirements.txt
mkdocs serve
```

构建：`mkdocs build`。
