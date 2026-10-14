#!/usr/bin/env python3
"""Checks the user-provided summaries and navigation in the 38 experiment pages."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "source" / "abstract.md").read_text(encoding="utf-8")
MATCHES = list(re.finditer(r"(?m)^[ \t]*### 实验[^\n]*", SOURCE))
assert len(MATCHES) == 38, f"expected 38 summaries, found {len(MATCHES)}"
NAV = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
for i, match in enumerate(MATCHES, 1):
    end = MATCHES[i].start() if i < len(MATCHES) else len(SOURCE)
    expected = SOURCE[match.start():end].rstrip()
    path = ROOT / "docs" / "experiments" / f"{i:02d}.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith(expected), f"Summary mismatch: {path}"
    assert text.count("[↓ 跳转到本实验的详细论述](#experiment-details)") == 1, path
    assert text.count('<a id="experiment-details"></a>') == 1, path
    assert text.count("[↑ 返回本页开头的实验总结](#)") == 1, path
    assert text.index("## 实验详细论述") > len(expected), path
    assert f"experiments/{i:02d}.md" in NAV, f"Missing from nav: {path}"
print("PASS: 38 unchanged summaries, detailed-section links, back-links and nav entries.")
