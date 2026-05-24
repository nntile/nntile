# C++ style — reference

Companion to [SKILL.md](SKILL.md). Read when applying rules to non-trivial
constructs or setting up tooling.

**Column width:** prose in this file uses at most **79 characters** per line.
Indented `cpp` / `yaml` code blocks may exceed 79 where needed for valid
syntax.

## Indentation and tabs

- **4 spaces** per indent level. **No tab characters** for indentation.
- Continuation lines for wrapped calls, `if` conditions, and expressions:
  indent **one level** (4 spaces) past the start of the statement, unless a
  clearer local pattern already exists in the file—then stay consistent
  **within that file**.

## Namespaces

Braces are **Allman** here too: `{` / `}` on their own lines.

```cpp
namespace project::detail
{
void helper();
}

namespace
{
void file_local();
}
```

- **Do not** indent the entire file for an outer `namespace` wrapper unless the
  codebase already does; if a file uses a single outer namespace, either no
  extra indent inside it or one level—**match surrounding files**.
- **Closing names** (e.g. `} // namespace detail`) are optional; if used, keep
  the comment short and within 79 columns.

## Classes and structs

### Brace and semicolon

```cpp
class Widget
{
    // ...
};

struct Point
{
    int x;
    int y;
};
```

### Access order

Default order (**top to bottom**): **`public`**, then **`protected`**, then
**`private`**. Omit empty sections.

### Member order (within each access section)

1. Nested types / type aliases (`using`, `typedef`)
2. Static constants (including `static constexpr`)
3. Constructors, destructor, assignment operators
4. Instance methods (group overloads together)
5. Static methods
6. Data members (static before instance, or match file convention—stay
   consistent)

`struct` defaults to public; still use explicit `public:` / `private:` when
mixing.

### `= default` / `= delete`

Place on the **same line** as the declaration if it fits within 79 columns;
otherwise break the signature first (parameters one per line), then keep
`= default` on the last line of the declaration.

## Includes

1. **Associated header** (e.g. `foo.cpp` includes `foo.hpp` first).
2. Blank line.
3. **Project** headers.
4. Blank line.
5. **Third-party** headers.
6. Blank line.
7. **Standard library** headers (use `<cstdint>`-style C headers in C++).

Use `#pragma once` or include guards—**match the project**. Guards, if used,
wrap the whole file; the `#endif` comment must fit in 79 columns.

## `const`, pointers, references

- Prefer **east const**: use `T const`, `T const&`, and `T const*` so `const`
  applies to what is on its left.
- **References**: keep `*` / `&` consistent in the file. Pick either `T *p` /
  `T &r` or `T* p` / `T& r`; default to `T *p` / `T &r` (space before `*` /
  `&`) if undecided.

## Control flow

### `if` / `for` / `while`

Space after keyword: `if (`, `for (`, `while (`.

`else` / `catch` on a **new line**, with Allman braces:

```cpp
if (cond)
{
    ...
}
else
{
    ...
}
```

### `switch`

```cpp
switch (tag)
{
case Kind::A:
    handle_a();
    break;
case Kind::B:
{
    int scoped = 1;
    handle_b(scoped);
    break;
}
default:
    break;
}
```

Braces around a `case` body when it needs local variables or multi-line logic;
`break` / `return` / `throw` explicit.

### Long conditions

Break **before** `&&` or `||` so the continuation line **starts** with the
operator (Black-like readability):

```cpp
if (first_condition
    && second_condition
    && third_condition)
{
    ...
}
```

Keep `(` on the line with `if`; indent continuation consistently (often 4
spaces past `if`).

## Constructor initializer lists

Colon starts the list; if it does not fit in 79 columns, put the colon at the
end of the constructor line and **one member per line**, **comma at end of
line**:

```cpp
MyClass::MyClass(
    Arg1 a1,
    Arg2 a2,
)
    : base_(std::move(a1)),
      member_(a2)
{
}
```

If the constructor name line is too long, wrap parameters using the same rules
as function calls; then the `:` line as above.

## Templates

If `template<typename ...>` or the declaration exceeds 79 characters, break
after `template<`, **one template parameter per line**, closing `>` on its own
line if needed:

```cpp
template<
    typename T,
    typename Allocator,
>
class Vector
{
    ...
};
```

Apply the same **call-wrapping** rules to function template arguments at call
sites.

## Lambdas

- Prefer **named** lambdas or local structs when the body is large or reused.
- For a multi-line lambda body, use **Allman** `{` / `}`.
- If captures or parameters overflow 79 columns, break after `(` of the
  parameter list (treat like a small function).

## `using` and type aliases

One primary name per alias statement; if the right-hand side is too long,
break after `=` with continuation indented.

## `clang-format` (canonical file)

The **source of truth** is the repository root **`.clang-format`**. Run
`clang-format` from the tree root (or ensure editors discover that file).

`clang-format` cannot match Black exactly; that profile approximates **79
columns**, **Allman**, and **fewer packed arguments**. Tune the repo root
**`.clang-format`** if the team refines rules.

**Key options** (see root file for full list):

- `ColumnLimit: 79`, `BreakBeforeBraces: Allman`, `IndentWidth: 4`,
  `UseTab: Never`
- `BinPackArguments: false`, `BinPackParameters: false`,
  `AlignAfterOpenBracket: DontAlign`
- `BreakConstructorInitializers: AfterColon` (line-ending commas on inits;
  switch to `BeforeComma` for leading-comma style)

**Notes:**

- `AllowShortBlocksOnASingleLine`, `AllowShortIfStatementsOnASingleLine`, and
  `AllowShortLoopsOnASingleLine` use boolean `false` (not `Never`) so the YAML
  parses across `clang-format` versions.
- `AfterColon` tends to put **one initializer per line** with commas at **line
  ends** (closer to Black-style trailing commas than `BeforeComma`, which uses
  leading commas on the next line). Prefer `BeforeComma` if the team wants
  leading commas.
- Optional on newer `clang-format` for **east const** (verify with your
  version): uncomment the `QualifierAlignment` / `QualifierOrder` lines at the
  top of **`.clang-format`** — drop if unsupported or noisy.
- Review diffs: some versions pack differently; **project consistency** beats
  blind tooling.

## Preprocessor

- Prefer wrapping long `#define` bodies with backslash-newline; each
  continuation line obeys 79 columns.
- Indent code inside `#if 0` / feature macros consistently with the rest of
  the file.
