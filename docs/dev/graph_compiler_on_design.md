# Graph compiler O(N) design

**Status:** active redesign  
**Branch:** `graph_api` / agent work on dense mapping  
**Related:** [graph_compile_perf_mnist.md](graph_compile_perf_mnist.md),
[torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md)

## Goal

One continuous `TensorGraph` that accumulates tensors and ops, compiled
incrementally into a continuous `TileGraph`, then lowered into StarPU via
`Runtime`, with **strict linear complexity**:

```text
extend TensorGraph (+1 tensor or +1 op)     : O(1) amortized
compile sealed slice of M tensor ops         : O(M + tiles_created)
recompile already-compiled slice             : forbidden (watermark skip)
extend TileGraph (+1 tile or +1 tile op)     : O(1) amortized
lower TileGraph slice of K tile ops → StarPU : O(K)
resolve TensorNode → tiles → handle (batch)  : O(#tiles)
```

N / M / K always mean work touched **this call**, not full session history.

Every op keeps the **general tiled** `lower_to_tile` path (arbitrary grid).
Per-op constant-factor shortcuts are out of scope for this redesign.

## Layers

```text
TensorGraph  (symbolic tensors + tensor ops)
     │  seal_phase + append_tensor_graph_phase
     ▼
TileGraph    (tiles + tile ops)
     │  Runtime::compile + execute_range
     ▼
Runtime      (StarPU payloads + submit)
```

torch_nntile owns one session-scoped triple
(`TensorGraph` / `TileGraph` / `Runtime`) and advances watermarks each step.

## Identity and indexing

- Graph topology keeps **pointers** (`TensorNode*`, `TileNode*`) on ops.
- `NodeId` is a **monotonic append-only index** (`0 .. next_id_-1`).
- Side tables are `std::vector` indexed by `NodeId` (amortized O(1) grow).
- After `gc_unmarked_data_nodes()`, holes are allowed (empty slots). IDs are
  never rebuilt mid-session (that would be O(session)).

## Mapping path (must be O(1) per node, O(#tiles) total)

```text
TensorNode.id  ──► tiles_by_id_[id]     → vector<TileNode*>
TensorNode.id  ──► layout_by_id_[id]    → TensorAxisLayout
TensorNode.id  ──► desc_by_id_[id]      → TileGraph::TensorDescriptor*
TileNode       ──► payload_             → shared_ptr<core::Tile> (StarPU)
```

Hot-path bridges **must not** use `std::map` / `std::set` keyed by pointer.

| Former structure | Replacement |
|------------------|-------------|
| `TensorNodeToTileMap` (`std::map`) | dense vector by `TensorNode::id()` |
| `TileGraph::tensors_by_source_` | dense vector by source `id()` |
| `TensorGraphTiling::layouts_` | dense vector by `id()` |
| `tensor_layout_fp` (`std::map` + string) | `uint64_t` hash on binding / layout |
| `collect_phase_tensors` (`std::set`) | generation stamp on `TensorNode` |
| `Runtime::tile_map_` | `TileNode::payload_` field |

## Incremental compile watermarks

| Watermark | Role |
|-----------|------|
| `TensorGraph::phase_seal_cursor_` | last sealed tensor op |
| tile op count at last append | last lowered tensor→tile |
| `Runtime::compiled_graph_op_count_` | last DCE / allocate |
| `Runtime::executed_op_end_` | last submitted |

Rules:

1. `data()` / `add_op()` only append.
2. Seal + append + `compile()` only touch `[cursor, end)`.
3. After `wait()`, `drop_all_ops()` keeps the SCATTER prefix (O(phase)); do
   not re-lower compacted history. When every compiled tile op has finished,
   also `Runtime::drop_fully_executed_history()` + `TileGraph::clear_ops()`
   so `execution_order_` does not retain session history.
4. Under `TORCH_NNTILE_SKIP_STARPU=1`, still call `execute_range` so the
   execute watermark advances (otherwise compile becomes O(session)).

## Runtime

- Allocate into `TileNode::payload_` (null until first allocate / adopt).
- `get_tile<T>(node)` is a typed cast of `payload_` (no hash lookup).
- Last-consumer reclaim builds dying lists **only for the pending suffix**
  (`tiles_dying_after_op_.size() == pending`, indexed via
  `tiles_dying_op_base_`). Never `assign(|execution_order_|, {})` — that
  alone made `runtime.compile` grow linearly with session tile-op count.
- Sparse last-consumer map over pending inputs only (not
  `vector[max_tile_id]` — tile ids are session-monotonic).
- After a full `wait()`, torch_nntile may drop `execution_order_` and
  `TileGraph::ops` (`drop_fully_executed_history` + `clear_ops`) so
  retained tile-op history does not grow with step count. Tile nodes and
  payloads stay.

## Out of scope (later)

- Per-op `lower_to_tile` micro-optimizations and buffer/node pooling.
- Compile-once + replay with scalar lifting (Adam `lr` / `num_iter`).
- MPI / `home_node` / `execution.json` schema changes.

## Validation

Script: `torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py`

```bash
STARPU_WORKERS_NOBIND=1 TORCH_NNTILE_SKIP_STARPU=1 \
  python torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py \
    --steps 500 --batch-size 100 --seed 42 --device nntile \
    --ncpu 1 --restrict-cpu --skip-accuracy-floor
```

Success: ms/step flat across 100→1000 steps (±5%); short StarPU accuracy
run still meets the ≥0.97 floor.
