---
name: Integrate Tensor Ops
overview: Add logical/compiled graph support for every tensor operation in `include/nntile/tensor`, including shape/attribute handling, execution dispatch, and minimal tests/docs so the full op set is usable through the graph APIs.
todos:
  - id: inventory-ops
    content: Catalog tensor ops + attrs + dtype support
    status: pending
  - id: logical-graph
    content: Add OpTypes/attrs/builders for all ops
    status: pending
  - id: compiled-graph
    content: Add executors + dtype validation + dispatch
    status: pending
  - id: tests-docs
    content: Add smoke tests + update graph.md
    status: pending
---

# Integrate Tensor Ops Into Graph

## Approach

- Inventory all tensor ops in [`include/nntile/tensor`](include/nntile/tensor) (exclude utility headers like `tensor.hh`, `traits.hh`, `distributions.hh`) and map each op to:
  - required inputs/outputs
  - required attributes (alpha/beta/axis/etc.)
  - expected output shape rules
  - supported dtypes (from explicit instantiations in `src/tensor/*.cc`)

## Logical graph updates

- Extend [`include/nntile/graph/logical_graph.hh`](include/nntile/graph/logical_graph.hh):
  - add `OpType` entries for each tensor op
  - add attribute structs (grouped by op family when fields are identical)
  - add these to `OpAttrs`
- Update `op_type_to_string()` in [`src/graph/logical_graph.cc`](src/graph/logical_graph.cc) and tests in [`tests/graph/logical_graph.cc`](tests/graph/logical_graph.cc) to cover the expanded enum.
- Implement builder functions in [`include/nntile/graph/logical_graph_ops.hh`](include/nntile/graph/logical_graph_ops.hh) and [`src/graph/logical_graph_ops.cc`](src/graph/logical_graph_ops.cc) mirroring tensor signatures, including:
  - shape inference and validation for each op
  - graph ownership + dtype checks (shared helpers)
  - in-place/accumulation wiring where outputs are also inputs

Pattern to follow for builder wiring:

```164:223:src/graph/logical_graph_ops.cc
// ...
OpAttrs attrs = GeluAttrs{};
x.graph().add_op(
    OpType::GELU,
    attrs,
    {&x},
    {&output}
);
// ...
```

## Compiled graph execution

- Add executor declarations in [`include/nntile/graph/compiled_graph_ops.hh`](include/nntile/graph/compiled_graph_ops.hh) and implement in [`src/graph/compiled_graph_ops.cc`](src/graph/compiled_graph_ops.cc):
  - dispatch by `DataType` to `nntile::tensor::*` kernels
  - handle mixed-dtype ops (e.g., `embedding`, `total_sum_accum`, `mask_scalar`) by validating input dtype combinations and dispatching on the primary value type
  - include any op-specific attrs (axis, batch_ndim, padding/stride/dilation, seed/mean/stddev, ignore_index, etc.)
- Update [`src/graph/compiled_graph.cc`](src/graph/compiled_graph.cc):
  - extend `validate_operation_data_types()` to each op’s supported dtype set
  - add `execute_op` dispatch for all new `OpType` values

## Tests + docs

- Add logical-graph tests in [`tests/graph/logical_graph_ops.cc`](tests/graph/logical_graph_ops.cc) for each op family (elementwise, broadcast/fiber, reduction, indexing, conv/embedding/optimizer, etc.) to assert op type, attrs, and shape wiring.
- Add compiled-graph smoke tests in [`tests/graph/compiled_graph_ops.cc`](tests/graph/compiled_graph_ops.cc) for representative ops per family; for GPU-only ops (e.g. flash SDPA), add guarded tests or compile-only coverage so CPU-only CI still passes.
- Update [`graph.md`](graph.md) to reflect the full op list and any dtype constraints.

## Notes/assumptions

- Map graph ops to the non-async tensor functions (the compiled graph already blocks on `execute()` + `wait()`).
- Utility headers without tensor kernels (`tensor.hh`, `traits.hh`, `distributions.hh`) are excluded unless you want graph wrappers for them explicitly.