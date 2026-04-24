---
name: cpp-coding-style
description: >-
  C++ style: 79 columns, Allman braces, 4-space indent, Black-like wrapping,
  includes, class layout, east const, final newline. Use for C++ edits,
  reviews, refactors.
---

# C++ coding style (Black-inspired)

Apply these rules when writing, editing, or reviewing C++ in this project.
The repo root **`.clang-format`** is the canonical formatter profile; keep it
aligned with this skill when either changes. For **template** / **switch**
detail and edge cases, see [reference.md](reference.md).

## Hard limits

- **Line length**: at most **79 characters** per line (including spaces and
  punctuation). Prefer breaking earlier at a natural boundary when it reads
  better.
- **Final newline**: every source file must end with a **newline** (POSIX
  end-of-line), not EOF immediately after the last character.

## Indentation

- **4 spaces** per indent level; **no tab characters** for indentation.
- Wrapped expressions: indent **one level** past the start of the statement
  unless the file already uses a consistent alternative—then **do not mix**
  styles in one file.

## Braces (curly brackets)

Use **Allman** style: `{` and `}` each on their **own line** for `if`, `for`,
`while`, `else`, `switch`, `try`/`catch`, `struct`, `class`, `enum class`,
**named and anonymous `namespace`**, and function definitions.

```cpp
void foo()
{
    if (condition)
    {
        body();
    }
    else
    {
        other();
    }
}
```

Multi-line lambdas use Allman `{` / `}`; if the signature overflows 79
columns, break parameters like a normal function (see
[reference.md](reference.md)).

## Namespaces

Opening `{` on its own line after the namespace header; closing `}` on its own
line. Do not add an extra file-wide indent level for a single outer namespace
**unless** the rest of the tree already does—**match neighboring files**.

## Classes and structs

- **`class` / `struct`**: name on one line, `{` on the next, closing `};` with
  `}` on its own line.
- **Access**: order sections **`public` → `protected` → `private`**; omit
  empty sections.
- **Members** (within each section): nested types / `using`, then static
  constants, then ctors / dtor / assignment, then methods (group overloads),
  then static methods, then data members. See [reference.md](reference.md) for
  `struct` / `= default` details.

## Includes

Order: **associated header** (in `.cpp`), blank line, **project** headers,
blank line, **third-party**, blank line, **standard library** (`<…>`). Use
`#pragma once` or guards **as the project already does**.

## `const`, pointers, references

- Prefer **east const** (`T const`, `T const&`, `T const*`).
- **Spacing**: keep `*` and `&` styling **consistent in the file** (e.g.
  `T *p`, `T const &r`).

## Long comments

If a comment would exceed 79 characters: split into **multiple** `//` lines
(or wrapped `/* */` lines), respecting word boundaries when possible.

## Long function calls (Black-like)

1. **`(` stays** on the same line as the callee (including `->` / `.` chains).
2. First argument on the **next** line, indented one level.
3. If still too long: **one argument per line**, **trailing commas** between
   arguments, **`)` on its own line**.

```cpp
result = some_very_long_function_name(
    first_argument,
    second_argument,
    third_argument,
);
```

Apply the same rules **recursively** inside arguments. **Templates**
(`make_shared`, etc.): same—`(` after the full callee, then break (see
[reference.md](reference.md) for multi-line `template<>`).

## Long expressions (`if`, `while`, chains)

Break **before** `&&` / `||` so the continuation line **starts with the
operator**. Keep `(` on the `if` / `while` line; indent continuations
consistently.

## Constructor initializer lists

If the list does not fit: put **`:`** after the closing `)` of the ctor
parameter list (or at the end of a wrapped ctor line), then **one base/member
init per line**, **comma at end of line**, then Allman `{` on the next line
after the list.

## “Black spirit” (general)

- Prefer vertical layout over dense one-liners near the limit.
- One logical statement per line; spaces around binary operators and after
  commas in single-line lists.
- Use **`nullptr`**, not `0` / `NULL`, for null pointers in new code unless
  constrained by legacy APIs.

## Checklist before finishing an edit

- [ ] No line longer than 79 characters.
- [ ] 4 spaces; no tabs for indent.
- [ ] Long comments split; long calls, conditions, ctor inits wrapped as
      above.
- [ ] Allman `{` / `}` for control flow, classes, namespaces, functions.
- [ ] Includes grouped and ordered; associated header first in `.cpp`.
- [ ] Class access / member order follows this skill.
- [ ] East const; consistent `*` / `&` spacing within the file.
- [ ] File ends with a newline.
