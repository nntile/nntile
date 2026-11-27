<!--
@copyright (c) 2026-present Skolkovo Institute of Science and Technology
                            (Skoltech), Russia. All rights reserved.

NNHaul is a task-based dynamic execution engine over N memory nodes:
a virtual planner builds one plan per batch from a drained snapshot,
and a literal executor issues exactly the planned operations on the
lanes of the node inventory.

@file docs/README.md
Documentation index: the chunk model, the trace manual, and a worked
trace example.

@version 1.0.0
-->

<!-- SPDX-License-Identifier: MIT -->

# NNHaul documentation

| Document | Contents |
|---|---|
| [../README.md](../README.md) | Overview, architecture, build and install, knobs, tests. |
| [chunk-model.md](chunk-model.md) | The chunk model in one page: one chunk class, tile decomposition, tails, what a codelet receives. Standalone. |
| [planner-api.md](planner-api.md) | The public planner interface: inputs, emitted plan, loading (`NNHAUL_PLANNER`), and a worked example planner. |
| [traces.md](traces.md) | The trace manual: lanes, Chrome Trace Event JSON format, enabling, clocks, overhead, analyzers and viewers. |
| [examples/](examples/) | A worked trace (`nnhaul-demo.json`) and its offline HTML report. |

The analyzer and viewers referenced by the trace manual live in
[../tools/](../tools/).

The kernel-side derivation for classic NNTile ops lives in the NNTile
tree; [chunk-model.md](chunk-model.md) is the standalone contract this
documentation set relies on.
