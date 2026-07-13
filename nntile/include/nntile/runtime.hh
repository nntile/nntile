/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/runtime.hh
 * ``nntile::Runtime`` — compile/execute a ``TileGraph`` (IR only).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Third-party headers
#include <starpu.h>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/core/execution_schedule.hh>
#include <nntile/dtype.hh>
#include <nntile/nn/graph_decl.hh>
#include <nntile/tensor/graph_data_node.hh>
#include <nntile/tensor/tensor_graph_tiling.hh>
#include <nntile/tile/graph_data_node.hh>
#include <nntile/tile/graph_decl.hh>
#include <nntile/tile/lowering_context.hh>
#include <nntile/core/tile.hh>

namespace nntile
{

//! StarPU-backed executor for a TileGraph (IR is separate).
class Runtime
{
  public:
    using TileNode = TileGraph::TileNode;
    using OpNode = TileGraph::OpNode;

    explicit Runtime(const TileGraph &graph);

    void compile();

    //! Submit ops [op_begin, op_end) asynchronously (no StarPU drain).
    //! After each submitted op, last-consumer tiles are invalidated via
    //! ``invalidate_submit`` and payload cleared (async w.r.t.
    //! already-submitted consumers). Call ``wait()`` to join StarPU before
    //! host readout.
    //! If ``submit_tasks`` is false, skip ``OpNode::execute`` (no StarPU
    //! inserts) but still advance the executed watermark and last-consumer
    //! reclaim — required so incremental ``compile()`` stays O(pending)
    //! under dry-run profiling.
    void execute_range(
        size_t op_begin,
        size_t op_end,
        bool submit_tasks = true);

    size_t execution_op_count() const { return execution_order_.size(); }

    //! Exclusive end index of the last op in ``[op_begin, op_end)`` that
    //! lists ``tile`` as an input. Returns ``op_begin`` if none.
    size_t last_input_consumer_end(
        TileNode const *tile,
        size_t op_begin,
        size_t op_end) const;

    //! Bind host data to a logical tensor or scatter to its tiles.
    template <typename T>
    void bind_data(
        TensorGraph::TensorNode const *tensor, const T *data, size_t count);

    template <typename T>
    void bind_data(
        TensorGraph::TensorNode const *tensor, const std::vector<T> &data);

    //! Bind via ``NNGraph::TensorNode`` (same as ``tensor->data()``).
    template <typename T>
    void bind_data(
        NNGraph::TensorNode const *tensor, const T *data, size_t count);

    template <typename T>
    void bind_data(
        NNGraph::TensorNode const *tensor, const std::vector<T> &data);

    //! Bind host data to a standalone tile (no tensor descriptor).
    template <typename T>
    void bind_data(TileNode const *tile, const T *data, size_t count);

    template <typename T>
    void bind_data(TileNode const *tile, const std::vector<T> &data);

    //! Submit all compiled ops asynchronously (no StarPU drain).
    //! Last-consumer ``invalidate_submit`` runs during submit (see
    //! ``execute_range``). Call ``wait()`` to join StarPU. Same submit
    //! contract as ``execute_range(0, execution_op_count())``.
    void execute();

    //! StarPU worker for ``STARPU_EXECUTE_ON_WORKER`` during tile op execution,
    //! or -1 for default StarPU placement. Set by ``execute()`` /
    //! ``execute_range()`` from the static execution schedule when one is
    //! installed.
    int starpu_worker_hint() const noexcept { return starpu_worker_hint_; }

    //! Block until all tasks submitted by ``execute()`` / ``execute_range()``
    //! have finished. Last-consumer tile invalidation already ran during
    //! submit; this only drains StarPU.
    void wait();

    //! Read a logical tensor or tile buffer marked for host I/O (input or
    //! output), same visibility rules as ``bind_data``.
    template <typename T>
    std::vector<T> get_output(TensorGraph::TensorNode const *tensor);

    template <typename T>
    std::vector<T> get_output(NNGraph::TensorNode const *tensor);

    template <typename T> std::vector<T> get_output(TileNode const *tile);

    template <typename T>
    nntile::core::Tile<T> &get_tile(const TileNode *node);

    DataType get_dtype(TensorGraph::TensorNode const *tensor) const;

    DataType get_dtype(NNGraph::TensorNode const *tensor) const;

    DataType get_dtype(const TileNode *node) const { return node->dtype(); }

    bool is_compiled() const { return compiled_; }

    //! Whether host data was copied into tiles for this logical tensor.
    bool is_initialized(TensorGraph::TensorNode const *tensor) const;

    bool is_initialized(NNGraph::TensorNode const *tensor) const;

    //! Clear initialized flag after staging readout (tile handle stays live).
    void invalidate_initialized(TensorGraph::TensorNode const *tensor);

    void invalidate_initialized(NNGraph::TensorNode const *tensor);

    //! Drop StarPU buffers for a logical tensor that is no longer marked
    //! input/output. No-op if still marked, unknown, or never allocated.
    void invalidate_logical_tiles(
        TensorGraph::TensorNode const *logical);

    //! Mark logical tensor tiles as host-populated (after acquire write I/O).
    void mark_initialized(TensorGraph::TensorNode const *tensor);

    //! Snapshot initialized tile buffers keyed by logical tensor (incremental reset).
    void export_initialized_tiles(
        std::unordered_map<TensorGraph::TensorNode const *,
            std::vector<std::shared_ptr<void>>> &out) const;

    //! Snapshot every allocated tile buffer keyed by logical tensor.
    void export_all_tiles(
        std::unordered_map<TensorGraph::TensorNode const *,
            std::vector<std::shared_ptr<void>>> &out) const;

    //! Map saved tile buffers onto new tile nodes; returns adopted tensors.
    std::vector<TensorGraph::TensorNode const *> stage_persisted_tiles(
        std::unordered_map<TensorGraph::TensorNode const *,
            std::vector<std::shared_ptr<void>>> const &persisted,
        TensorNodeToTileMap const &tile_map);

    void restore_persisted_init_state(
        std::unordered_map<TensorGraph::TensorNode const *, bool> const
            &persisted_init);

    ExecutionSchedule const &execution_schedule() const
    {
        return execution_schedule_;
    }

    bool has_execution_schedule() const
    {
        return !execution_schedule_.ops.empty();
    }

    //! After ``compile()``: build round-robin schedule from DCE order (does not
    //! write a file; use ``generate_round_robin_execution_json`` for that).
    ExecutionSchedule generate_round_robin_execution_schedule() const;

    //! After ``compile()``: batch-slice affinity tile split (same JSON schema).
    ExecutionSchedule generate_affinity_batch_execution_schedule() const;

    //! Optional: pin workers during ``execute()`` (from generator or
    //! ``load_execution_schedule_json``). Without a schedule, StarPU chooses
    //! workers at runtime.
    void set_execution_schedule(ExecutionSchedule schedule);

    void load_execution_schedule(std::string const &path);

    //! Load ``execution.json`` once (per path), then ``set_execution_schedule``.
    //! Re-reads only when ``path`` changes. Clears cache if apply fails.
    void apply_execution_schedule_from_file(std::string const &path);

    void clear_execution_schedule_file_cache();

    //! ``compile()`` then round-robin schedule in memory (convenience for tests).
    void compile_with_round_robin_schedule();

    void write_execution_schedule_json(std::string const &path) const;

  private:
    friend class NNGraph;
    friend void compile_incremental_nn_phase(
        FinishedTensorPhase const &,
        NNGraph &,
        TensorGraphTiling const &,
        TileGraph &,
        Runtime &,
        TileGraphIncrementalState &,
        TensorNodeToTileMap &,
        bool,
        std::unordered_map<TensorGraph::TensorNode const *,
            std::vector<std::shared_ptr<void>>> const *,
        std::unordered_map<TensorGraph::TensorNode const *, bool> const *);

    void allocate_missing_tiles();
    void eliminate_dead_ops();
    void build_tile_last_consumer_map();
    void sync_tile_marks_from_logical();
    void release_dead_tiles_after_op(size_t op_idx);
    void queue_dead_tiles_after_op(size_t op_idx);
    void flush_queued_dead_tiles();
    void invalidate_tile_buffer(
        const TileNode *node,
        const std::shared_ptr<void> &tile_ptr);
    void require_compiled() const;
    bool tensor_requires_init_at_execute(
        TileGraph::TensorDescriptor const &desc) const;
    void validate_initialized_inputs_at_compile();

    template <typename T, typename NntileT, typename CastT>
    void bind_data_impl(const TileNode *node, const T *data, size_t count);
    template <typename T, typename NntileT, typename CastT>
    void get_output_impl(const TileNode *node, std::vector<T> &result);

    const TileGraph &graph_;
    //! Payloads live on ``TileNode::payload_`` (O(1) field access).
    std::vector<std::shared_ptr<OpNode>> execution_order_;
    ExecutionSchedule execution_schedule_;
    std::optional<ExecutionSchedule> execution_schedule_file_cache_;
    std::string execution_schedule_file_cache_path_;
    bool compiled_ = false;
    int starpu_worker_hint_ = -1;
    std::unordered_map<TensorGraph::TensorNode const *, bool> init_state_;
    std::unordered_map<const TileNode *, std::shared_ptr<void>> tile_adoption_;
    std::unordered_set<const TileNode *> live_tile_nodes_;
    //! ``tiles_dying_after_op_[i]`` = tiles whose last consumer is op ``i``.
    //! Built in O(pending); reclaim is O(#dying at i), not O(#all tiles).
    std::vector<std::vector<const TileNode *>> tiles_dying_after_op_;
    //! Scratch for last-consumer tiles; flushed via invalidate_submit during
    //! ``execute_range`` (not deferred to ``wait()``).
    std::vector<const TileNode *> queued_dead_tiles_;
    //! Highest exclusive op index already run via execute / execute_range.
    size_t executed_op_end_ = 0;
    //! How many ``graph_.ops()`` entries have been appended into
    //! ``execution_order_``. Incremental ``compile()`` only pulls ops beyond
    //! this watermark so cost stays O(pending) rather than O(history).
    size_t compiled_graph_op_count_ = 0;
    //! How many ``graph_.tile_nodes()`` have been considered for allocation.
    //! Ingress may lower marked staging tiles before any new op is appended;
    //! those nodes must still be allocated without scanning full history.
    size_t compiled_tile_node_count_ = 0;
};

} // namespace nntile

#include <nntile/nn/graph_data_node.hh>

namespace nntile
{

// ---------------------------------------------------------------------------
// Runtime template implementation
// ---------------------------------------------------------------------------

namespace tile_bind_detail
{

inline bool tensor_desc_has_input_tile(TileGraph::TensorDescriptor const &d)
{
    for (TileGraph::TileNode *t : d.tiles)
    {
        if (t != nullptr && t->is_input())
        {
            return true;
        }
    }
    return false;
}

inline bool tensor_desc_has_output_tile(TileGraph::TensorDescriptor const &d)
{
    for (TileGraph::TileNode *t : d.tiles)
    {
        if (t != nullptr && t->is_output())
        {
            return true;
        }
    }
    return false;
}

inline bool use_logical_layout(TileGraph::TensorDescriptor const *desc,
    TensorGraph::TensorNode const *tensor)
{
    if (desc == nullptr || tensor == nullptr || desc->source_node != tensor)
    {
        return false;
    }
    return desc->tiles.size() > static_cast<size_t>(1) ||
           (desc->tiles.size() == static_cast<size_t>(1) &&
               desc->tiles[0]->name() != tensor->name());
}

} // namespace tile_bind_detail

namespace tile_detail
{

template <typename T> struct dtype_for
{
    static_assert(sizeof(T) == 0,
        "Unsupported tile type for get_tile; use fp32_t, fp64_t, "
        "fp16_t, bf16_t, int64_t, bool_t, or fp32_fast_* variants");
};

template <> struct dtype_for<nntile::fp32_t>
{
    static constexpr DataType value = DataType::FP32;
};
template <> struct dtype_for<nntile::fp32_fast_tf32_t>
{
    static constexpr DataType value = DataType::FP32_FAST_TF32;
};
template <> struct dtype_for<nntile::fp32_fast_fp16_t>
{
    static constexpr DataType value = DataType::FP32_FAST_FP16;
};
template <> struct dtype_for<nntile::fp32_fast_bf16_t>
{
    static constexpr DataType value = DataType::FP32_FAST_BF16;
};
template <> struct dtype_for<nntile::fp64_t>
{
    static constexpr DataType value = DataType::FP64;
};
template <> struct dtype_for<nntile::fp16_t>
{
    static constexpr DataType value = DataType::FP16;
};
template <> struct dtype_for<nntile::bf16_t>
{
    static constexpr DataType value = DataType::BF16;
};
template <> struct dtype_for<nntile::int64_t>
{
    static constexpr DataType value = DataType::INT64;
};
template <> struct dtype_for<nntile::bool_t>
{
    static constexpr DataType value = DataType::BOOL;
};

} // namespace tile_detail

namespace tile_layout_io
{

//! Decode a flat offset into tile-local coordinates (C-order: last dim
//! stride 1).
inline void tile_linear_to_index(Index linear_offset,
    const std::vector<Index> &shape,
    std::vector<Index> &index)
{
    const size_t ndim = shape.size();
    index.resize(ndim);
    if (ndim == 0)
    {
        return;
    }
    std::vector<Index> stride(ndim);
    stride[ndim - 1] = 1;
    for (size_t i = ndim - 1; i > 0; --i)
    {
        stride[i - 1] = stride[i] * shape[i];
    }
    Index rem = linear_offset;
    for (size_t i = 0; i < ndim - 1; ++i)
    {
        const Index div = rem / stride[i];
        rem -= div * stride[i];
        index[i] = div;
    }
    index[ndim - 1] = rem;
}

//! Dense offset matching bind_data / get_output flat layout (C-order).
inline Index dense_linear_index(
    const std::vector<Index> &shape, const std::vector<Index> &global_coord)
{
    if (shape.size() != global_coord.size())
    {
        throw std::invalid_argument(
            "dense_linear_index: shape/coord size mismatch");
    }
    Index idx = 0;
    Index stride = 1;
    for (size_t d = shape.size(); d-- > 0;)
    {
        const Index g = global_coord[d];
        if (g < 0 || g >= shape[d])
        {
            throw std::out_of_range(
                "dense_linear_index: global coord OOB");
        }
        idx += g * stride;
        stride *= shape[d];
    }
    return idx;
}

[[deprecated("Use tile_linear_to_index")]]
inline void fortran_tile_linear_to_index(Index linear_offset,
    const std::vector<Index> &shape,
    std::vector<Index> &index)
{
    tile_linear_to_index(linear_offset, shape, index);
}

[[deprecated("Use dense_linear_index")]]
inline Index fortran_dense_linear_index(
    const std::vector<Index> &shape, const std::vector<Index> &global_coord)
{
    return dense_linear_index(shape, global_coord);
}

template <typename T, typename NntileT, typename CastT>
void scatter_logical_tensor(const TensorAxisLayout &lay,
    const std::vector<TileGraph::TileNode *> &tiles,
    const T *host,
    size_t count,
    Runtime &rt)
{
    Index nelems = 1;
    for (Index s : lay.tensor_shape())
    {
        nelems *= s;
    }
    if (count != static_cast<size_t>(nelems))
    {
        throw std::runtime_error(
            "Runtime::bind_data: dense size mismatch for logical "
            "tensor");
    }
    const Index vol = lay.grid_volume();
    if (static_cast<Index>(tiles.size()) != vol)
    {
        throw std::runtime_error(
            "Runtime::bind_data: tile vector size mismatch");
    }
    std::vector<Index> gc;
    std::vector<Index> local;
    std::vector<Index> global;
    for (Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, gc);
        const std::vector<Index> ts = lay.tile_shape_at(gc);
        Index tne = 1;
        for (Index v : ts)
        {
            tne *= v;
        }
        TileGraph::TileNode *tn = tiles[static_cast<size_t>(lin)];
        auto &tile = rt.template get_tile<NntileT>(tn);
        auto tile_local = tile.acquire(STARPU_W);
        for (Index lf = 0; lf < tne; ++lf)
        {
            tile_linear_to_index(lf, ts, local);
            lay.global_coord(gc, local, global);
            const Index di =
                dense_linear_index(lay.tensor_shape(), global);
            tile_local[lf] =
                NntileT(static_cast<CastT>(host[static_cast<size_t>(di)]));
        }
        tile_local.release();
    }
}

template <typename T, typename NntileT, typename CastT>
void gather_logical_tensor(const TensorAxisLayout &lay,
    const std::vector<TileGraph::TileNode *> &tiles,
    std::vector<T> &out,
    Runtime &rt)
{
    Index nelems = 1;
    for (Index s : lay.tensor_shape())
    {
        nelems *= s;
    }
    out.resize(static_cast<size_t>(nelems));
    const Index vol = lay.grid_volume();
    std::vector<Index> gc;
    std::vector<Index> local;
    std::vector<Index> global;
    for (Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, gc);
        const std::vector<Index> ts = lay.tile_shape_at(gc);
        Index tne = 1;
        for (Index v : ts)
        {
            tne *= v;
        }
        TileGraph::TileNode *tn = tiles[static_cast<size_t>(lin)];
        const auto &tile = rt.template get_tile<NntileT>(tn);
        auto tile_local = tile.acquire(STARPU_R);
        for (Index lf = 0; lf < tne; ++lf)
        {
            tile_linear_to_index(lf, ts, local);
            lay.global_coord(gc, local, global);
            const Index di =
                dense_linear_index(lay.tensor_shape(), global);
            out[static_cast<size_t>(di)] =
                static_cast<T>(static_cast<CastT>(tile_local[lf]));
        }
        tile_local.release();
    }
}

} // namespace tile_layout_io

template <typename T>
nntile::core::Tile<T> &Runtime::get_tile(const TileNode *node)
{
    if (node == nullptr || !node->has_payload())
    {
        throw std::runtime_error(
            "Runtime::get_tile: node not found");
    }
    if (node->dtype() != tile_detail::dtype_for<T>::value)
    {
        throw std::runtime_error(
            "Runtime::get_tile: wrong type (requested type does "
            "not match tile dtype)");
    }
    auto ptr = std::static_pointer_cast<nntile::core::Tile<T>>(
        node->payload());
    return *ptr;
}

template <typename T>
void Runtime::bind_data(
    TensorGraph::TensorNode const *tensor, const T *data, size_t count)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::bind_data: tensor must be non-null");
    }
    const TileGraph::TensorDescriptor *desc =
        graph_.get_tensor_descriptor(tensor);
    if (desc == nullptr || desc->source_node != tensor)
    {
        throw std::runtime_error(
            "Runtime::bind_data: tensor has no TileGraph "
            "descriptor (lower with source_node set)");
    }
    const TensorGraphTiling *tsch = graph_.tiling_scheme();
    const bool use_logical =
        tsch != nullptr &&
        tile_bind_detail::use_logical_layout(desc, tensor);
    if (use_logical)
    {
        const TensorAxisLayout *lay = tsch->find(desc->source_node);
        if (lay == nullptr)
        {
            throw std::runtime_error(
                "Runtime::bind_data: missing tiling for tensor '" +
                tensor->name() + "'");
        }
        if (!tile_bind_detail::tensor_desc_has_input_tile(*desc) &&
            !tile_bind_detail::tensor_desc_has_output_tile(*desc))
        {
            throw std::runtime_error("bind_data: mark_input(true) or "
                                     "mark_output(true) on tensor '" +
                                     tensor->name() + "'");
        }
        switch (desc->dtype)
        {
        case DataType::FP32:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::fp32_t,
                float>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP32_FAST_TF32:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::fp32_fast_tf32_t,
                float>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP32_FAST_FP16:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::fp32_fast_fp16_t,
                float>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP32_FAST_BF16:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::fp32_fast_bf16_t,
                float>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP64:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::fp64_t,
                double>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::FP16:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::fp16_t,
                float>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::BF16:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::bf16_t,
                float>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::INT64:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::int64_t,
                std::int64_t>(*lay, desc->tiles, data, count, *this);
            break;
        case DataType::BOOL:
            tile_layout_io::scatter_logical_tensor<T,
                nntile::bool_t,
                bool>(*lay, desc->tiles, data, count, *this);
            break;
        default:
            throw std::runtime_error(
                "Runtime::bind_data: unsupported dtype for "
                "logical tensor '" +
                tensor->name() + "'");
        }
        mark_initialized(tensor);
        return;
    }
    if (desc->tiles.empty())
    {
        throw std::runtime_error(
            "Runtime::bind_data: descriptor has no tiles");
    }
    TileNode const *tnode = desc->tiles[0];
    if (tnode == nullptr || !tnode->has_payload())
    {
        throw std::runtime_error(
            "Runtime::bind_data: tile storage not allocated");
    }
    if (!tnode->is_input() && !tnode->is_output())
    {
        throw std::runtime_error(
            "bind_data: tile '" + tnode->name() +
            "' must be marked as input or output on the data node");
    }
    DataType dtype = tnode->dtype();
    switch (dtype)
    {
    case DataType::FP32:
        bind_data_impl<T, nntile::fp32_t, float>(tnode, data, count);
        break;
    case DataType::FP32_FAST_TF32:
        bind_data_impl<T, nntile::fp32_fast_tf32_t, float>(tnode, data, count);
        break;
    case DataType::FP32_FAST_FP16:
        bind_data_impl<T, nntile::fp32_fast_fp16_t, float>(tnode, data, count);
        break;
    case DataType::FP32_FAST_BF16:
        bind_data_impl<T, nntile::fp32_fast_bf16_t, float>(tnode, data, count);
        break;
    case DataType::FP64:
        bind_data_impl<T, nntile::fp64_t, double>(tnode, data, count);
        break;
    case DataType::FP16:
        bind_data_impl<T, nntile::fp16_t, float>(tnode, data, count);
        break;
    case DataType::BF16:
        bind_data_impl<T, nntile::bf16_t, float>(tnode, data, count);
        break;
    case DataType::INT64:
        bind_data_impl<T, nntile::int64_t, std::int64_t>(tnode, data, count);
        break;
    case DataType::BOOL:
        bind_data_impl<T, nntile::bool_t, bool>(tnode, data, count);
        break;
    default:
        throw std::runtime_error("Unsupported data type for binding");
    }
    mark_initialized(tensor);
}

template <typename T>
void Runtime::bind_data(
    TensorGraph::TensorNode const *tensor, const std::vector<T> &data)
{
    bind_data(tensor, data.data(), data.size());
}

template <typename T>
void Runtime::bind_data(
    NNGraph::TensorNode const *tensor, const T *data, size_t count)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::bind_data: NN tensor must be non-null");
    }
    bind_data(tensor->data(), data, count);
}

template <typename T>
void Runtime::bind_data(
    NNGraph::TensorNode const *tensor, const std::vector<T> &data)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::bind_data: NN tensor must be non-null");
    }
    bind_data(tensor->data(), data);
}

template <typename T>
void Runtime::bind_data(
    TileNode const *tile, const T *data, size_t count)
{
    if (tile == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::bind_data: tile must be non-null");
    }
    if (!tile->has_payload())
    {
        throw std::runtime_error(
            "Runtime::bind_data: tile storage not allocated");
    }
    if (!tile->is_input() && !tile->is_output())
    {
        throw std::runtime_error(
            "bind_data: tile '" + tile->name() +
            "' must be marked as input or output on the data node");
    }
    DataType dtype = tile->dtype();
    switch (dtype)
    {
    case DataType::FP32:
        bind_data_impl<T, nntile::fp32_t, float>(tile, data, count);
        break;
    case DataType::FP32_FAST_TF32:
        bind_data_impl<T, nntile::fp32_fast_tf32_t, float>(tile, data, count);
        break;
    case DataType::FP32_FAST_FP16:
        bind_data_impl<T, nntile::fp32_fast_fp16_t, float>(tile, data, count);
        break;
    case DataType::FP32_FAST_BF16:
        bind_data_impl<T, nntile::fp32_fast_bf16_t, float>(tile, data, count);
        break;
    case DataType::FP64:
        bind_data_impl<T, nntile::fp64_t, double>(tile, data, count);
        break;
    case DataType::FP16:
        bind_data_impl<T, nntile::fp16_t, float>(tile, data, count);
        break;
    case DataType::BF16:
        bind_data_impl<T, nntile::bf16_t, float>(tile, data, count);
        break;
    case DataType::INT64:
        bind_data_impl<T, nntile::int64_t, std::int64_t>(tile, data, count);
        break;
    case DataType::BOOL:
        bind_data_impl<T, nntile::bool_t, bool>(tile, data, count);
        break;
    default:
        throw std::runtime_error("Unsupported data type for binding");
    }
}

template <typename T>
void Runtime::bind_data(
    TileNode const *tile, const std::vector<T> &data)
{
    bind_data(tile, data.data(), data.size());
}

template <typename T, typename NntileT, typename CastT>
void Runtime::bind_data_impl(
    const TileNode *node, const T *data, size_t count)
{
    auto &tile = get_tile<NntileT>(node);
    if (count != static_cast<size_t>(tile.nelems))
    {
        throw std::runtime_error(
            "Data size mismatch for tile '" + node->name() + "'");
    }
    auto tile_local = tile.acquire(STARPU_W);
    for (size_t i = 0; i < count; ++i)
    {
        tile_local[i] = NntileT(static_cast<CastT>(data[i]));
    }
    tile_local.release();
}

template <typename T>
std::vector<T> Runtime::get_output(
    TensorGraph::TensorNode const *tensor)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::get_output: tensor must be non-null");
    }
    const TileGraph::TensorDescriptor *desc =
        graph_.get_tensor_descriptor(tensor);
    if (desc == nullptr || desc->source_node != tensor)
    {
        throw std::runtime_error(
            "Runtime::get_output: tensor has no TileGraph "
            "descriptor (lower with source_node set)");
    }
    const TensorGraphTiling *tsch = graph_.tiling_scheme();
    const bool use_logical =
        tsch != nullptr &&
        tile_bind_detail::use_logical_layout(desc, tensor);
    if (use_logical)
    {
        const TensorAxisLayout *lay = tsch->find(desc->source_node);
        if (lay == nullptr)
        {
            throw std::runtime_error(
                "Runtime::get_output: missing tiling for tensor '" +
                tensor->name() + "'");
        }
        if (!tile_bind_detail::tensor_desc_has_output_tile(*desc) &&
            !tile_bind_detail::tensor_desc_has_input_tile(*desc))
        {
            throw std::runtime_error(
                "get_output: tensor '" + tensor->name() +
                "' has no input/output tiles; call mark_input(true) or "
                "mark_output(true) on the tensor data node");
        }
        std::vector<T> result;
        switch (desc->dtype)
        {
        case DataType::FP32:
            tile_layout_io::gather_logical_tensor<T,
                nntile::fp32_t,
                float>(*lay, desc->tiles, result, *this);
            break;
        case DataType::FP32_FAST_TF32:
            tile_layout_io::gather_logical_tensor<T,
                nntile::fp32_fast_tf32_t,
                float>(*lay, desc->tiles, result, *this);
            break;
        case DataType::FP32_FAST_FP16:
            tile_layout_io::gather_logical_tensor<T,
                nntile::fp32_fast_fp16_t,
                float>(*lay, desc->tiles, result, *this);
            break;
        case DataType::FP32_FAST_BF16:
            tile_layout_io::gather_logical_tensor<T,
                nntile::fp32_fast_bf16_t,
                float>(*lay, desc->tiles, result, *this);
            break;
        case DataType::FP64:
            tile_layout_io::gather_logical_tensor<T,
                nntile::fp64_t,
                double>(*lay, desc->tiles, result, *this);
            break;
        case DataType::FP16:
            tile_layout_io::gather_logical_tensor<T,
                nntile::fp16_t,
                float>(*lay, desc->tiles, result, *this);
            break;
        case DataType::BF16:
            tile_layout_io::gather_logical_tensor<T,
                nntile::bf16_t,
                float>(*lay, desc->tiles, result, *this);
            break;
        case DataType::INT64:
            tile_layout_io::gather_logical_tensor<T,
                nntile::int64_t,
                std::int64_t>(*lay, desc->tiles, result, *this);
            break;
        case DataType::BOOL:
            tile_layout_io::gather_logical_tensor<T,
                nntile::bool_t,
                bool>(*lay, desc->tiles, result, *this);
            break;
        default:
            throw std::runtime_error(
                "Runtime::get_output: unsupported dtype for "
                "logical tensor '" +
                tensor->name() + "'");
        }
        return result;
    }
    if (desc->tiles.empty())
    {
        throw std::runtime_error(
            "Runtime::get_output: descriptor has no tiles");
    }
    TileNode const *tnode = desc->tiles[0];
    if (tnode == nullptr || !tnode->has_payload())
    {
        throw std::runtime_error(
            "Runtime::get_output: tile storage not allocated");
    }
    if (!tnode->is_output() && !tnode->is_input())
    {
        throw std::runtime_error(
            "get_output: tile '" + tnode->name() +
            "' is not marked as input or output; call mark_input(true) or "
            "mark_output(true) on the data node");
    }
    DataType dtype = tnode->dtype();
    std::vector<T> result;
    switch (dtype)
    {
    case DataType::FP32:
        get_output_impl<T, nntile::fp32_t, float>(tnode, result);
        break;
    case DataType::FP32_FAST_TF32:
        get_output_impl<T, nntile::fp32_fast_tf32_t, float>(tnode, result);
        break;
    case DataType::FP32_FAST_FP16:
        get_output_impl<T, nntile::fp32_fast_fp16_t, float>(tnode, result);
        break;
    case DataType::FP32_FAST_BF16:
        get_output_impl<T, nntile::fp32_fast_bf16_t, float>(tnode, result);
        break;
    case DataType::FP64:
        get_output_impl<T, nntile::fp64_t, double>(tnode, result);
        break;
    case DataType::FP16:
        get_output_impl<T, nntile::fp16_t, float>(tnode, result);
        break;
    case DataType::BF16:
        get_output_impl<T, nntile::bf16_t, float>(tnode, result);
        break;
    case DataType::INT64:
        get_output_impl<T, nntile::int64_t, std::int64_t>(tnode, result);
        break;
    case DataType::BOOL:
        get_output_impl<T, nntile::bool_t, bool>(tnode, result);
        break;
    default:
        throw std::runtime_error("Unsupported data type for get_output");
    }
    return result;
}

template <typename T>
std::vector<T> Runtime::get_output(NNGraph::TensorNode const *tensor)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::get_output: NN tensor must be non-null");
    }
    return get_output<T>(tensor->data());
}

template <typename T>
std::vector<T> Runtime::get_output(TileNode const *tile)
{
    if (tile == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::get_output: tile must be non-null");
    }
    if (!tile->is_output() && !tile->is_input())
    {
        throw std::runtime_error(
            "get_output: tile must be marked input or output on the data node");
    }
    if (!tile->has_payload())
    {
        throw std::runtime_error(
            "Runtime::get_output: tile storage not allocated");
    }
    DataType dtype = tile->dtype();
    std::vector<T> result;
    switch (dtype)
    {
    case DataType::FP32:
        get_output_impl<T, nntile::fp32_t, float>(tile, result);
        break;
    case DataType::FP32_FAST_TF32:
        get_output_impl<T, nntile::fp32_fast_tf32_t, float>(tile, result);
        break;
    case DataType::FP32_FAST_FP16:
        get_output_impl<T, nntile::fp32_fast_fp16_t, float>(tile, result);
        break;
    case DataType::FP32_FAST_BF16:
        get_output_impl<T, nntile::fp32_fast_bf16_t, float>(tile, result);
        break;
    case DataType::FP64:
        get_output_impl<T, nntile::fp64_t, double>(tile, result);
        break;
    case DataType::FP16:
        get_output_impl<T, nntile::fp16_t, float>(tile, result);
        break;
    case DataType::BF16:
        get_output_impl<T, nntile::bf16_t, float>(tile, result);
        break;
    case DataType::INT64:
        get_output_impl<T, nntile::int64_t, std::int64_t>(tile, result);
        break;
    case DataType::BOOL:
        get_output_impl<T, nntile::bool_t, bool>(tile, result);
        break;
    default:
        throw std::runtime_error("Unsupported data type for get_output");
    }
    return result;
}

template <typename T, typename NntileT, typename CastT>
void Runtime::get_output_impl(
    const TileNode *node, std::vector<T> &result)
{
    auto &tile_buf = get_tile<NntileT>(node);
    result.resize(tile_buf.nelems);
    auto tile_local = tile_buf.acquire(STARPU_R);
    for (Index i = 0; i < tile_buf.nelems; ++i)
    {
        result[i] = static_cast<T>(static_cast<CastT>(tile_local[i]));
    }
    tile_local.release();
}

inline DataType Runtime::get_dtype(
    NNGraph::TensorNode const *tensor) const
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::get_dtype: NN tensor must be non-null");
    }
    return get_dtype(tensor->data());
}

inline bool Runtime::is_initialized(
    TensorGraph::TensorNode const *tensor) const
{
    if (tensor == nullptr)
    {
        return false;
    }
    auto it = init_state_.find(tensor);
    return it != init_state_.end() && it->second;
}

inline bool Runtime::is_initialized(NNGraph::TensorNode const *tensor) const
{
    if (tensor == nullptr)
    {
        return false;
    }
    return is_initialized(tensor->data());
}

inline void Runtime::invalidate_initialized(
    TensorGraph::TensorNode const *tensor)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::invalidate_initialized: tensor must be non-null");
    }
    init_state_[tensor] = false;
}

inline void Runtime::invalidate_initialized(NNGraph::TensorNode const *tensor)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "Runtime::invalidate_initialized: NN tensor must be non-null");
    }
    invalidate_initialized(tensor->data());
}

} // namespace nntile
