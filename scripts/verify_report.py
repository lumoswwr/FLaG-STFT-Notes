#!/usr/bin/env python3
"""Verify verbatim experiment summaries, navigation, image links and internal anchors."""
from pathlib import Path
from urllib.parse import unquote
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCE = (ROOT / "source/abstract.md").read_text(encoding="utf-8")
GROUPS = {
    "01-stft-window.md": [1, 2, 3, 4, 5, 11, 12, 14],
    "02-token-frequency.md": [6, 7, 8],
    "03-padding-reconstruction.md": [9, 10, 13, 15, 16, 17],
    "04-sprint.md": [18, 19, 20, 21, 22],
    "05-dc-bands.md": [23, 24, 25, 26],
    "06-mean-fusion.md": [27, 28, 29, 30, 31],
    "07-gate-projection.md": [32, 33, 34, 35],
    "08-simplified-pooling.md": [36, 37, 38],
}
matches = list(re.finditer(r"(?m)^[ \t]*### 实验[^\n]*", SOURCE))
assert len(matches) == 38, f"Expected 38 experiment summaries, got {len(matches)}"
summaries = {
    i + 1: SOURCE[match.start():(matches[i + 1].start() if i + 1 < len(matches) else len(SOURCE))].rstrip()
    for i, match in enumerate(matches)
}
all_ids = [n for ns in GROUPS.values() for n in ns]
assert sorted(all_ids) == list(range(1, 39)), "Each experiment must occur exactly once"
nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
for filename, ns in GROUPS.items():
    path = DOCS / "chapters" / filename
    content = path.read_text(encoding="utf-8")
    assert f"chapters/{filename}" in nav, f"Missing nav: {filename}"
    summary_section, sep, details = content.partition("\n---\n\n## 实验详细论述")
    assert sep and summary_section.index("## 实验总结") >= 0, path
    for n in ns:
        expected = (
            f'<a id="summary-{n:02d}"></a>\n\n'
            + summaries[n]
            + f'\n\n[查看本实验详细论述](#detail-{n:02d})'
        )
        assert expected in summary_section, f"Verbatim summary mismatch: experiment {n}"
        assert f'<a id="detail-{n:02d}"></a>' in details, f"Missing detail anchor {n}"
        assert f'[返回实验 {n} 的总结](#summary-{n:02d})' in details, f"Missing back-link {n}"
    assert "记录来源与代码核对" not in content, path
    assert "原始记录归档" not in content, path

assert not any((DOCS / "experiments").glob("*.md")), "Old one-experiment-per-file pages must be removed"
assert "experiments/01.md" not in nav, "Stale per-experiment navigation"

link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
for path in DOCS.rglob("*.md"):
    content = path.read_text(encoding="utf-8")
    for match in link_pattern.finditer(content):
        target = unquote(match.group(1).strip().strip("<>"))
        if target.startswith(("https://", "http://", "mailto:", "data:")):
            continue
        local, sep, fragment = target.partition("#")
        resolved = (path.parent / local).resolve() if local else path
        assert resolved.is_file(), f"Missing target: {path.relative_to(DOCS)} -> {target}"
        if sep and fragment.startswith(("summary-", "detail-")):
            dest = resolved.read_text(encoding="utf-8")
            assert f'id="{fragment}"' in dest, f"Missing anchor: {path} -> {target}"

print("PASS: 38 unchanged summaries in 8 chapters; navigation, return links, images and local links.")
