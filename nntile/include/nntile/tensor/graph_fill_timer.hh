/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/tensor/graph_fill_timer.hh
 * Reentrant wall timer for nntile record work (record(nntile)).
 *
 * @version 1.1.0
 */

#pragma once

namespace nntile
{

//! Accumulates wall time of nntile record work (aten kernels, views,
//! TensorGraph mutations, optimizer / loss pybind).
//!
//! Nested scopes on one thread do not double-count. Snapshot
//! ``seconds()`` around a Python record window; the delta is
//! ``record(nntile)``. Remaining record wall is PyTorch overhead
//! (``record(torch)``).
class GraphFillScope
{
  public:
    GraphFillScope();
    ~GraphFillScope();
    GraphFillScope(GraphFillScope const &) = delete;
    GraphFillScope &operator=(GraphFillScope const &) = delete;

    static double seconds();
};

//! ``TORCH_NNTILE_SKIP_KERNELS=1``: do not insert TensorGraph compute
//! ops. TensorNode / TensorRef construction still runs. Last-drop
//! ``UNREGISTER`` is still inserted (StarPU unregister task).
bool skip_tensor_graph_ops();

} // namespace nntile
