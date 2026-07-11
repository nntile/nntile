# Plan: bias support for `aten::linear` in `torch_nntile`

**Scope:** `torch_nntile` extension only (ATen PrivateUse1 / `device="nntile"`).  
**Status:** bias is **not** supported today; this document is the implementation plan.

---

## Current state

`F.linear` / `nn.Linear` on `device="nntile"` runs only the matmul path
(`tensor::gemm`). Bias is explicitly rejected:

| Site | Behavior |
|------|----------|
| `torch_nntile/csrc/nntile_linear.cpp` forward | `TORCH_CHECK(false, "nntile linear: bias is not supported")` if `bias` is set |
| same file, `linear_backward` | `TORCH_CHECK(!output_mask[2], ...)` — no `grad_bias` |
| `torch_nntile/README.md` | Documents `F.linear` / `nn.Linear` **(no bias)** → `tensor::gemm` |

**Workarounds already in tree (outside the ATen op):**

- `DeepReLU` uses `nn.Linear(..., bias=False)`.
- GPT-2 helpers add bias as a separate `out + bias` via `aten::add` broadcast
  (`NntileConv1D`, attention Q/K/V/O). That works but materializes a full
  broadcast (or `scale_slice` chain), instead of a fiber add.

**Reference already in `torch_nntile`:** `LayerNorm` affine bias uses the same
NNTile primitives we need:

- Forward: `nntile::tensor::add_fiber_inplace` along the feature axis  
  (`nntile_executor.cpp` → `tensor_layer_norm_forward_fp32`)
- Backward `grad_bias`: `nntile::tensor::sum_fiber`  
  (`tensor_layer_norm_backward_fp32`)

---

## PyTorch semantics to match

```text
aten::linear(input, weight, bias=None) -> Tensor
  input:  [..., in_features]
  weight: [out_features, in_features]
  bias:   [out_features]   # optional, 1-D
  out:    [..., out_features] = input @ weight.T (+ bias)

aten::linear_backward(self, grad_output, weight, output_mask)
  -> (grad_input, grad_weight, grad_bias)
```

Notes:

- Bias is **always** 1-D of length `weight.size(0)`. No general broadcast shapes.
- `linear_backward` does **not** receive the bias tensor; when
  `output_mask[2]` is true, allocate `grad_bias` with shape `[out_features]`
  (`weight.size(0)`).
- `grad_bias = sum(grad_output over all axes except the last)`.

---

## Preferred NNTile ops

| Role | Op | Why |
|------|----|-----|
| Forward bias add | `add_fiber_inplace` (or out-of-place `add_fiber`) | Bias is a **1-D fiber** along the last axis of `out` — same pattern as LayerNorm |
| Backward `grad_bias` | `sum_fiber` | Dual of fiber add; already used for LN `grad_bias` |

**Do not** use the generic `aten::add` + `scale_slice` broadcast path for the
fused linear bias: it is correct but more expensive and duplicates what GPT-2
already does manually.

`add_slice` is the wrong shape model here (slice rank = tensor rank − 1). Keep
it for LN mean centering / softmax backward, not linear bias.

Axis / batch conventions (match LayerNorm):

```text
axis       = output.dim() - 1          # last feature axis
batch_ndim = 0                         # all leading dims are reduced / broadcast
alpha = 1, beta = 1                    # out = 1 * bias + 1 * gemm_out
redux  = 0                             # same as kNormRedux in executor
```

For 1-D input (`[in_features]` → `[out_features]`), `axis = 0`; `sum_fiber` /
`add_fiber` still apply (no leading dims to reduce).

---

## Implementation steps

### 1. Validation (`nntile_linear.cpp`)

Replace the hard reject with checks when `bias` is present:

- nntile device, `float32`, contiguous
- `bias.dim() == 1` and `bias.size(0) == weight.size(0)`

Leave weight/input checks as they are.

### 2. Forward (`linear` / `linear.out`)

1. Keep existing `prepare_linear_operands` + `tensor_gemm_fp32` into `output`.
2. If bias is set:
   - `pin_graph_op_inputs` must include bias (extend the pin set used for gemm).
   - After gemm, call a small executor helper, e.g.
     `tensor_linear_add_bias_fp32(output, bias)`, that:
     - resolves graph nodes for `output` and `bias`
     - runs `add_fiber_inplace(1, bias_node, 1, output_node, axis, 0)`
3. Prefer **in-place** fiber add into the gemm result (one buffer), mirroring
   LayerNorm’s `copy` + `add_fiber_inplace` pattern — here gemm already wrote
   `output`, so only the in-place add is needed.

Stub / no-libnntile builds: keep a clear `require_libnntile` (or no-op stub)
consistent with other executor entry points.

### 3. Backward (`linear_backward`)

Existing `grad_input` / `grad_weight` gemm paths stay unchanged (bias does not
affect them).

When `output_mask[2]`:

1. `grad_bias = at::empty({weight.size(0)}, grad_output.options())` (or
   `empty({out_features}, ...)`).
2. `pin_graph_op_output(grad_bias, /*param-like*/ true)` as appropriate.
3. Executor helper `tensor_linear_grad_bias_fp32(grad_output, grad_bias)`:
   - `clear(grad_bias_node)` then
   - `sum_fiber(grad_out_node, grad_bias_node, axis, batch_ndim=0, redux=0, 1, 0)`
4. Register param grad for autograd / host copy the same way weight grads do
   (`register_param_grad_node` + `register_grad_alias_for_host_copy`) when a
   graph node exists.
5. Return `{grad_input, grad_weight, grad_bias}` instead of an undefined third
   tensor.

### 4. Executor API surface

Add to `nntile_executor.h` / `.cpp` (libnntile and stub):

```text
void tensor_linear_add_bias_fp32(at::Tensor &output, const at::Tensor &bias);
void tensor_linear_grad_bias_fp32(
    const at::Tensor &grad_output, at::Tensor &grad_bias);
```

Reuse includes already present for LayerNorm:
`add_fiber_inplace.hh`, `sum_fiber.hh`, `clear.hh`.

Keep helpers thin; do not fold bias into `tensor_gemm_fp32`.

### 5. Tests

Add `torch_nntile/tests/test_linear_bias_parity.py` (mirror existing
`test_deep_relu_parity` / `test_add_parity` style):

| Case | Check |
|------|--------|
| 2-D forward | `F.linear(x, w, b)` vs CPU |
| ND forward (`[B,S,H]`) | same |
| 1-D input | same |
| Backward | `grad_input`, `grad_weight`, `grad_bias` vs CPU |
| `bias=None` | still works (regression) |
| Bad bias shape | raises |

Tolerances: start with `rtol=1e-5, atol=1e-5` (float32 gemm).

### 6. Docs / optional cleanups (same or follow-up PR)

- Update `torch_nntile/README.md` table: drop “(no bias)”; note
  `gemm` + `add_fiber_inplace` / `sum_fiber`.
- Optional: switch `NntileConv1D` / attention bias adds to `F.linear(..., bias=)`
  once parity is green (behavior-equivalent, fewer graph nodes). Not required
  for correctness of the ATen op.

---

## Files to touch

| File | Change |
|------|--------|
| `torch_nntile/csrc/nntile_linear.cpp` | Validate bias; forward add; backward `sum_fiber` |
| `torch_nntile/csrc/nntile_executor.h` | Declare bias helpers |
| `torch_nntile/csrc/nntile_executor.cpp` | Implement helpers (+ stub) |
| `torch_nntile/tests/test_linear_bias_parity.py` | New parity tests |
| `torch_nntile/README.md` | Document supported bias |

No changes to core `nntile::tensor` APIs — reuse existing fiber ops.

---

## Acceptance criteria

1. `F.linear(x, w, b)` and `nn.Linear(..., bias=True)` on `device="nntile"`
   match CPU forward within float32 tolerance.
2. Autograd produces correct `grad_bias` (and existing grads still match).
3. `bias=None` path unchanged.
4. Graph recording pins bias / `grad_bias` and registers param grads for bias
   parameters (same pattern as weight / LN bias).
5. CPU-only CI: new pytest cases pass under existing `torch_nntile` test setup.

---

## Non-goals

- Fusing bias into a single custom StarPU gemm+bias kernel (fiber add after
  gemm is enough).
- Changing `addmm` (already has its own beta·self + alpha·mm path).
- MPI / multi-node distribution of bias.
- Non-float32 dtypes.
