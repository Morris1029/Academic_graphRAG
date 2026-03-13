"""简单乱码检查脚本（只校验，不修改文件）。

用法：
  python utils/check_mojibake.py
  python utils/check_mojibake.py path/to/file1 path/to/file2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# 常见 UTF-8/GBK 误解码后出现的碎片
SUSPECT_TOKENS = [
    "\u93c2",  # 鏂
    "\u934f",  # 鍏
    "\u9983",  # 馃
    "\u6d7c",  # 浼
    "\u7481",  # 璁
    "\u951b",  # 锛
    "\u951f",  # 锟
    "\u70d8",  # 烨/烘 等常见片段
    "\u7f01",  # 缁
    "\u9a9e",  # 骞
]

DEFAULT_FILES = [
    "models/constructor/kt_gen.py",
    "config/base_config.yaml",
]


def scan_file(path: Path, suspects: Iterable[str]) -> List[Tuple[int, str, List[str]]]:
    findings: List[Tuple[int, str, List[str]]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        hit = [token for token in suspects if token in line]
        if hit:
            findings.append((lineno, line.strip(), hit))
    return findings


def main() -> int:
    paths = [Path(p) for p in (sys.argv[1:] or DEFAULT_FILES)]
    has_issue = False

    for path in paths:
        print(f"[CHECK] {path}")
        if not path.exists():
            print("  - 文件不存在，跳过")
            continue
        findings = scan_file(path, SUSPECT_TOKENS)
        if not findings:
            print("  - OK: 未发现疑似乱码片段")
            continue

        has_issue = True
        print(f"  - FOUND: {len(findings)} 行疑似乱码")
        for lineno, content, hit in findings[:20]:
            print(f"    {lineno}: 命中 {hit} | {content}")

    return 1 if has_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())

