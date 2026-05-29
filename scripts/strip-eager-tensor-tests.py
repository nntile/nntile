#!/usr/bin/env python3
"""Remove eager nntile::core::tensor reference tests from graph tensor op tests."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def find_matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    i = open_pos
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        elif c in "\"'":
            quote = c
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    break
                i += 1
        i += 1
    raise ValueError("unbalanced braces")


def remove_check_functions(text: str) -> str:
    pattern = re.compile(
        r"(?:/\*![\s\S]*?\*/\s*)?"
        r"(?://[^\n]*\n\s*)*"
        r"template\s*<[^>]+>\s*"
        r"void\s+check_\w+_vs_tensor_api\s*\([^;]*\)\s*\{",
        re.MULTILINE,
    )
    while True:
        m = pattern.search(text)
        if not m:
            break
        start = m.start()
        brace = text.find("{", m.end() - 1)
        end = find_matching_brace(text, brace) + 1
        text = text[:start] + text[end:]
    return text


def remove_matches_test_cases(text: str) -> str:
    markers = list(re.finditer(r"\bTEST_CASE(?:_METHOD)?\s*\(", text))
    to_remove: list[tuple[int, int]] = []
    for m in markers:
        paren = text.find("(", m.end() - 1)
        depth = 0
        i = paren
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        header_end = i + 1
        while header_end < len(text) and text[header_end] in " \t\n":
            header_end += 1
        if header_end >= len(text) or text[header_end] != "{":
            continue
        header = text[m.start() : header_end + 1]
        if (
            "matches nntile::tensor::" not in header
            and "matches tensor API" not in header
            and "check_" not in header
        ):
            body_start = header_end
            brace = body_start
            body_end = find_matching_brace(text, brace) + 1
            body = text[body_start:body_end]
            if not re.search(r"check_\w+_vs_tensor_api", body):
                continue
        brace = header_end
        end = find_matching_brace(text, brace) + 1
        to_remove.append((m.start(), end))
    for start, end in reversed(to_remove):
        text = text[:start] + text[end:]
    return text


def clean_includes(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if "nntile/core/tensor" in line:
            continue
        out.append(line)
    return "".join(out)


def clean_header_comment(text: str) -> str:
    text = re.sub(
        r" against nntile::tensor::[a-zA-Z0-9_]+",
        "",
        text,
    )
    text = re.sub(
        r"Test TensorGraph ([a-z_]+) operation against\n"
        r" \* nntile::tensor::[a-z_]+\.",
        r"Test TensorGraph \1 operation.",
        text,
    )
    return text


def process_file(path: Path) -> bool:
    original = path.read_text()
    text = original
    text = remove_check_functions(text)
    text = remove_matches_test_cases(text)
    text = clean_includes(text)
    text = clean_header_comment(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "nntile/tests/tensor/ops")
    changed = 0
    for path in sorted(root.glob("*.cc")):
        if process_file(path):
            print(path.name)
            changed += 1
    print(f"updated {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
