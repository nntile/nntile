#!/usr/bin/env python3
"""Migrate TensorGraph::data / NNGraph::tensor old name-at-creation API."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def skip_raw_string(text: str, i: int) -> int:
    m = re.match(r'R"([^()\\]{0,16})\(', text[i:])
    if not m:
        raise ValueError("bad raw string")
    delim = m.group(1)
    i += m.end()
    end_pat = f'){delim}"'
    end = text.find(end_pat, i)
    if end < 0:
        raise ValueError("unterminated raw string")
    return end + len(end_pat)


def skip_string_literal(text: str, i: int) -> int:
    if text.startswith('R"', i):
        return skip_raw_string(text, i)
    quote = text[i]
    i += 1
    while i < len(text):
        c = text[i]
        if c == "\\" and quote != "'":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    raise ValueError("unterminated string")


def skip_non_code(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        if text[i : i + 2] == "//":
            i = text.find("\n", i)
            if i < 0:
                return n
            continue
        if text[i : i + 2] == "/*":
            end = text.find("*/", i + 2)
            if end < 0:
                return n
            i = end + 2
            continue
        if text[i] in " \t\n\r":
            i += 1
            continue
        break
    return i


def parse_paren_call(text: str, open_paren: int) -> tuple[list[str], int]:
    assert text[open_paren] == "("
    depth_paren = 1
    depth_brace = 0
    depth_bracket = 0
    i = open_paren + 1
    start = i
    args: list[str] = []
    n = len(text)
    in_string: str | None = None
    while i < n:
        c = text[i]
        if in_string is not None:
            if in_string == "R":
                ni = skip_raw_string(text, i)
                i = ni
                in_string = None
                continue
            if c == "\\" and in_string != "'":
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in '"\'':
            if text.startswith('R"', i):
                in_string = "R"
            else:
                in_string = c
            i += 1
            continue
        if c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1
            if depth_paren == 0:
                tail = text[start:i].strip()
                if tail:
                    args.append(tail)
                return args, i + 1
        elif c == "{":
            depth_brace += 1
        elif c == "}":
            depth_brace -= 1
        elif c == "[":
            depth_bracket += 1
        elif c == "]":
            depth_bracket -= 1
        elif (
            c == ","
            and depth_paren == 1
            and depth_brace == 0
            and depth_bracket == 0
        ):
            args.append(text[start:i].strip())
            start = i + 1
        i += 1
    raise ValueError("unclosed paren")


def is_string_literal(s: str) -> bool:
    t = s.strip()
    return t.startswith('"') or t.startswith('R"')


TENSOR_DATA_RECEIVERS = frozenset(
    {
        "tensor_graph",
        "tensor_graph_",
        "tg_graph",
        "tg_src",
        "g_ref",
        "g_tile",
        "tg",
        # TensorGraph locals in tests (never TileGraph ``graph`` / ``g_tile``).
        # Do not list bare ``g``: TileGraph parity tests use ``TileGraph g``.
        "other",
        "graph2",
    }
)


def is_tensor_test_file(path: Path) -> bool:
    p = str(path).replace("\\", "/")
    return "tests/graph/tensor/" in p


def should_migrate_data(receiver: str, path: Path) -> bool:
    r = receiver.strip()
    if r in TENSOR_DATA_RECEIVERS:
        return True
    if is_tensor_test_file(path) and r == "graph":
        return True
    return False


def should_migrate_tensor(receiver: str) -> bool:
    return receiver.strip() in (
        "graph",
        "g",
        "graph_",
        "nn_graph",
        "nng",
        "other",
    )


def recv_before_dot(text: str, dot_idx: int) -> tuple[str, int]:
    k = dot_idx - 1
    while k >= 0 and text[k] in " \t":
        k -= 1
    end = k + 1
    while k >= 0 and (text[k].isalnum() or text[k] == "_"):
        k -= 1
    start = k + 1
    return text[start:end], start


def find_method_calls(
    text: str,
) -> list[tuple[int, int, str, str, str, int]]:
    """yield call_start, after, receiver, sep (. or ->), method, open_paren."""
    out: list[tuple[int, int, str, str, str, int]] = []
    i = 0
    n = len(text)
    while i < n:
        i = skip_non_code(text, i)
        if i >= n:
            break
        if text.startswith('R"', i) or text[i] in '"\'':
            i = skip_string_literal(text, i)
            continue
        if text.startswith("->", i):
            recv, recv_start = recv_before_dot(text, i)
            rest = text[i:]
            for meth in ("data", "tensor"):
                pref = f"->{meth}("
                if rest.startswith(pref):
                    open_paren = i + len(pref) - 1
                    try:
                        _args, after = parse_paren_call(text, open_paren)
                    except ValueError:
                        i += 1
                        break
                    out.append(
                        (
                            recv_start,
                            after,
                            recv,
                            "->",
                            meth,
                            open_paren,
                        ),
                    )
                    i = after
                    break
            else:
                i += 1
            continue
        if text[i] == ".":
            recv, recv_start = recv_before_dot(text, i)
            rest = text[i:]
            for meth in ("data", "tensor"):
                pref = f".{meth}("
                if rest.startswith(pref):
                    open_paren = i + len(pref) - 1
                    try:
                        _args, after = parse_paren_call(text, open_paren)
                    except ValueError:
                        i += 1
                        break
                    out.append(
                        (
                            recv_start,
                            after,
                            recv,
                            ".",
                            meth,
                            open_paren,
                        ),
                    )
                    i = after
                    break
            else:
                i += 1
            continue
        i += 1
    return out


def migrate_data(receiver: str, sep: str, args: list[str]) -> str | None:
    if (
        len(args) == 3
        and is_string_literal(args[1])
        and "DataType::" in args[2]
    ):
        shape, name, dtype = args
        inner = f"{receiver}{sep}data({shape}, {dtype})->set_name({name})"
        return f"&{inner}"
    if len(args) == 2 and is_string_literal(args[1]):
        shape, name = args
        if name.strip() == '""':
            return f"{receiver}{sep}data({shape})"
        inner = f"{receiver}{sep}data({shape})->set_name({name})"
        return f"&{inner}"
    return None


def migrate_tensor(receiver: str, sep: str, args: list[str]) -> str | None:
    if len(args) == 4 and is_string_literal(args[1]):
        shape, name, dtype, rg = args
        return (
            f"{receiver}{sep}tensor({shape}, {dtype}, {rg})->set_name({name})"
        )
    if (
        len(args) == 3
        and is_string_literal(args[1])
        and "DataType::" in args[2]
    ):
        shape, name, dtype = args
        return f"{receiver}{sep}tensor({shape}, {dtype})->set_name({name})"
    return None


def process_file(path: Path, dry: bool) -> bool:
    text = path.read_text()
    calls = find_method_calls(text)
    repls: list[tuple[int, int, str]] = []
    for call_start, after, receiver, sep, meth, open_paren in reversed(calls):
        args, _ = parse_paren_call(text, open_paren)
        old = text[call_start:after]
        new_text = None
        if meth == "data" and should_migrate_data(receiver, path):
            new_text = migrate_data(receiver, sep, args)
        elif meth == "tensor" and should_migrate_tensor(receiver):
            new_text = migrate_tensor(receiver, sep, args)
        if new_text is not None and new_text != old:
            repls.append((call_start, after, new_text))
    if not repls:
        return False
    out: list[str] = []
    pos = 0
    for start, end, new in sorted(repls):
        out.append(text[pos:start])
        out.append(new)
        pos = end
    out.append(text[pos:])
    nt = "".join(out)
    if not dry:
        path.write_text(nt)
    return True


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    paths = [Path(p) for p in argv if not p.startswith("-")]
    if not paths:
        roots = [
            Path("tests/graph/tensor"),
            Path("examples"),
            Path("tests/graph/tile/append_tensor_graph_phase.cc"),
            Path("tests/graph/tile/tile_graph.cc"),
            Path("tests/graph/tile/add_mixed_tile.cc"),
            Path("tests/graph/nn"),
            Path("tests/graph/module"),
            Path("tests/graph/model"),
        ]
        paths = []
        for r in roots:
            if r.is_file():
                paths.append(r)
            elif r.is_dir():
                paths.extend(sorted(r.rglob("*.cc")))
    changed: list[Path] = []
    for p in paths:
        if process_file(p, dry):
            changed.append(p)
    for p in changed:
        print("migrated" + (" (dry)" if dry else ""), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
