# execution.json schema

Static **task schedule** for graph `Runtime::execute()`. This file describes
which **logical worker** runs each tile op and how tiles are **virtually** split.
It does **not** describe StarPU memory homes, NUMA placement, or MPI ownership.

## Workflow

1. Build the graph and call `Runtime::compile()` (DCE only; StarPU may schedule
   tasks at runtime if no static schedule is set).
2. **Optional — generate:** `generate_round_robin_execution_schedule()` or
   `generate_round_robin_execution_json(...)`.
3. **Optional:** inspect or edit `execution.json`.
4. **Load:** `load_execution_schedule_json(path)` then `set_execution_schedule()`.
5. `execute()` — each op runs on the assigned worker (`STARPU_EXECUTE_ON_WORKER`).

Regenerate the file when the compiled op list changes (graph edit, different tiling).

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `policy` | string | Generator id, e.g. `round_robin_virtual_tensor_split` or `affinity_batch_virtual_tensor_split` |
| `hardware` | object | `num_workers`, `worker_kind` (`cpu` or `cuda`) |
| `schedule_fingerprint` | object | `op_count`, `op_names[]` — must match compile |
| `virtual_tile_workers` | array | Virtual tile → worker map (debugging) |
| `ops` | array | Per-op schedule entries |

## `schedule_fingerprint`

```json
"schedule_fingerprint": {
  "op_count": 42,
  "op_names": ["TILE_GEMM", "TILE_ADD", "..."]
}
```

`Runtime::set_execution_schedule` rejects loads when `op_count` or `op_names` do not
match the current compiled execution order.

## `virtual_tile_workers[]`

```json
{ "tile": "model_...__t0", "virtual_worker": 0 }
```

Round-robin rule: tile grid index `lin` → `worker = lin % num_workers`.

## `ops[]`

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Position in post-DCE `execution_order` (0-based) |
| `op` | string | `TileGraph::OpNode::op_name()` |
| `name` | string | Task label (may be empty) |
| `worker` | int | Logical worker id for this op |
| `writable_tiles` | string[] | Output / in-place tile names |
| `read_tiles` | string[] | Read-only input tile names |

## Round-robin op assignment

1. Single writable output → worker owning that output tile.
2. Multiple writable / in-place outputs → worker with largest total writable
   byte volume among candidates (tie → lower worker id).

## Example (minimal)

```json
{
  "policy": "round_robin_virtual_tensor_split",
  "hardware": { "num_workers": 2, "worker_kind": "cpu" },
  "schedule_fingerprint": { "op_count": 1, "op_names": ["TILE_ADD"] },
  "virtual_tile_workers": [
    { "tile": "z", "virtual_worker": 0 }
  ],
  "ops": [
    {
      "index": 0,
      "op": "TILE_ADD",
      "name": "TILE_ADD@1",
      "worker": 0,
      "writable_tiles": ["z"],
      "read_tiles": ["x", "y"]
    }
  ]
}
```

## Related

- [graph_static_execution_plan.md](graph_static_execution_plan.md)
- API: `nntile/include/nntile/core/execution_schedule.hh`
