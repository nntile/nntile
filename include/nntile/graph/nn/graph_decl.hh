/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/graph_decl.hh
 * NNGraph class declaration (included by graph.hh).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tensor/graph.hh>
#include <nntile/graph/tensor/tensor_graph_tiling.hh>
#include <nntile/graph/tile/graph_decl.hh>
#include <nntile/graph/tile/lowering_context.hh>

namespace nntile::graph
{

class Runtime;

namespace module
{
class Module;
}

struct NNGraphExecState;
struct TileGraphIncrementalState;

//! User-facing graph with autograd; wraps a forward-only ``TensorGraph``.
class NNGraph
{
    friend class module::Module;

  public:
    //! Tensor node - full definition in graph_data_node.hh
    class TensorNode;

    //! Op node (AutoGradFunction) - full definition in graph_op_node.hh
    class OpNode;

    //! Destructor defined in .cc (needs complete TensorNode for unique_ptr)
    ~NNGraph();

  private:
    std::string name_;
    TensorGraph tensor_graph_;
    std::vector<std::unique_ptr<TensorNode>> tensors_;
    std::vector<std::shared_ptr<OpNode>> op_nodes_;

  public:
    explicit NNGraph(const std::string &name = "");

    // -----------------------------------------------------------------
    // Tensor Creation
    // -----------------------------------------------------------------

    TensorNode *tensor(std::vector<Index> shape,
        DataType dtype = DataType::FP32,
        bool requires_grad = true);

    TensorNode *tensor(
        TensorGraph::TensorNode *data, bool requires_grad = false);

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    const std::string &name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return op_nodes_.size(); }

    //! Find autograd wrapper for a tensor-graph data node (linear scan).
    TensorNode *get_tensor(TensorGraph::TensorNode const *data);
    const TensorNode *get_tensor(TensorGraph::TensorNode const *data) const;

    std::vector<std::string> tensor_names() const;

    //! Flat parameter tensors (same order as ``named_parameters()``).
    //! Lazily rebuilt from **all** registered ``Module`` roots on this graph
    //! (``parent_ == nullptr``); qualified names match
    //! ``named_parameters_recursive`` on each root subtree.
    std::vector<TensorNode *> parameters() const;

    //! Named parameters for every root subtree (lazy rebuild with
    //! ``parameters()``).
    std::vector<std::pair<std::string, TensorNode *>> named_parameters() const;

    const std::vector<std::unique_ptr<TensorNode>> &tensors() const
    {
        return tensors_;
    }
    const std::vector<std::shared_ptr<TensorGraph::OpNode>> &ops() const
    {
        return tensor_graph_.ops();
    }

    TensorGraph &tensor_graph() { return tensor_graph_; }
    const TensorGraph &tensor_graph() const { return tensor_graph_; }

    // -----------------------------------------------------------------
    // Phase finish + implicit tile lowering (owns TileGraph / Runtime state)
    // -----------------------------------------------------------------

    //! Seal ops since the last phase; stash the slice for
    //! ``lower_and_compile``; optionally clear NN autograd for the next phase.
    //! Tensor identity is by pointer; string names need not be unique across
    //! iterations.
    FinishedTensorPhase finish_phase(bool reset_autograd_state = true);

    //! Lower the last ``finish_phase`` slice with \p tiling, reuse tile
    //! buffers when layouts match, append archive metadata.  Requires
    //! ``finish_phase`` since the previous ``lower_and_compile`` (or
    //! construction).
    void lower_and_compile(TensorGraphTiling const &tiling);

    //! Same using ``TensorGraphTiling::from_tensor_graph(tensor_graph())``.
    void lower_and_compile();

    //! StarPU executor after ``lower_and_compile`` (bind / execute / outputs).
    Runtime &runtime();

    //! Whether ``lower_and_compile`` has run at least once.
    bool has_runtime() const;

    friend void compile_incremental_nn_phase(
        FinishedTensorPhase const &exec_phase,
        NNGraph &nn_graph_for_suffix,
        TensorGraphTiling const &tiling,
        TileGraph &tile_graph,
        Runtime &runtime,
        TileGraphIncrementalState &state,
        TensorNodeToTileMap &tile_map,
        bool archive_phase);

    //! Delegate to ``TensorGraph::seal_phase()`` (persistent = input/output).
    TensorGraph::PhaseSnapshot seal_phase();

    //! Explicit carried list mapped to tensor-graph data nodes.
    TensorGraph::PhaseSnapshot seal_phase(
        std::vector<TensorNode const *> const &carried);

    void reset_phase_seal_cursor();

    size_t phase_seal_cursor() const
    {
        return tensor_graph_.phase_seal_cursor();
    }

    //! Clear op_nodes_ (used when retain_graph=false after backward)
    void clear_op_nodes();

    //! Set producer_=nullptr on all tensors that had producers
    void clear_producers_on_tensors();

    // -----------------------------------------------------------------
    // Gradient helpers
    // -----------------------------------------------------------------

    bool requires_grad(const TensorNode *tensor) const;
    void set_requires_grad(TensorNode *tensor, bool value = true);

    //! Get or create gradient tensor. Does NOT add CLEAR.
    //! Returns (grad_tensor, is_first_write): is_first_write is true when
    //! the gradient was just created, so the caller should use overwrite
    //! (beta=0); false means accumulate (beta=1).
    std::pair<TensorNode *, bool> get_or_create_grad(
        TensorNode *tensor, const std::string &grad_name);

    //! Register op for backward. Stores op only when grad mode enabled
    //! and any input requires grad. Sets producer on outputs.
    void register_op(std::shared_ptr<OpNode> op);

    // -----------------------------------------------------------------
    // Gradient recording mode (no_grad)
    // -----------------------------------------------------------------

    //! Check if gradient recording is enabled for this graph
    bool is_grad_enabled() const { return grad_enabled_; }

    //! Set gradient recording (for testing; prefer NoGradGuard)
    void set_grad_enabled(bool enabled) { grad_enabled_ = enabled; }

    //! RAII guard to temporarily disable gradient recording.
    //! Use: { auto g = graph.no_grad(); ... } or NNGraph::NoGradGuard
    //! guard(&graph);
    class NoGradGuard
    {
      public:
        explicit NoGradGuard(NNGraph *graph);
        ~NoGradGuard();
        NoGradGuard(const NoGradGuard &) = delete;
        NoGradGuard &operator=(const NoGradGuard &) = delete;

      private:
        NNGraph *graph_;
        bool prev_;
    };

    //! Create a guard that disables grad recording until scope exit
    NoGradGuard no_grad() { return NoGradGuard(this); }

    // -----------------------------------------------------------------
    // Dynamic graph naming (multi-forward on one NNGraph)
    // -----------------------------------------------------------------

    //! When non-empty, ``module::Module::tensor_name`` appends ``_`` + tag so
    //! repeated forwards on the same graph create distinct underlying
    //! ``TensorGraph::TensorNode`` names when needed.  User tensors created
    //! via
    //! ``NNGraph::tensor`` with an explicit ``name`` are unchanged unless
    //! their producer uses ``Module::tensor_name``.  Disables
    //! ``enable_auto_tensor_name_phase_suffix``.
    void set_tensor_name_suffix_tag(std::string tag);

    //! Drop the suffix tag so later ``Module::tensor_name`` calls use only
    //! ``moduleName_localName``.  Disables automatic phase suffix mode.
    void clear_tensor_name_suffix_tag();

    std::string const &tensor_name_suffix_tag() const
    {
        return tensor_name_suffix_tag_;
    }

    //! Use ``_0``, ``_1``, … suffixes for ``Module`` tensor names; advanced by
    //! ``lower_and_compile``.  Call after ``finish_phase``.
    void enable_auto_tensor_name_phase_suffix(bool enable = true);

    bool auto_tensor_name_phase_suffix_enabled() const
    {
        return auto_tensor_name_phase_suffix_;
    }

    //! Advance auto suffix after ``lower_and_compile`` (low-level hook when
    //! calling ``compile_incremental_nn_phase`` directly).
    void bump_auto_tensor_name_phase_suffix_after_compile();

    // -----------------------------------------------------------------
    // Phase archives (implicit history after lowering)
    // -----------------------------------------------------------------

    //! Append one archive entry (``compile_incremental_nn_phase`` does this
    //! when
    //! ``archive_phase`` is true).
    void push_tensor_phase_archive(TensorPhaseArchiveEntry entry);

    //! Lowered phases in order (tensor slice + tile op range).
    std::vector<TensorPhaseArchiveEntry> const &tensor_phase_archives() const
    {
        return tensor_phase_archives_;
    }

    //! Drop archive entries (e.g. long runs).
    void clear_tensor_phase_archives();

    // -----------------------------------------------------------------
    // String representation
    // -----------------------------------------------------------------

    std::string to_string() const;

    std::string to_mermaid() const { return tensor_graph_.to_mermaid(); }

  private:
    void clear_pending_compile_if_same(FinishedTensorPhase const &compiled);

    void ensure_exec_state();

    void register_live_module(module::Module *mod);

    void unregister_live_module(module::Module *mod);

    void mark_module_parameter_cache_dirty();

    void ensure_module_parameter_cache() const;

    void rebuild_module_parameter_cache() const;

    //! Parent that lists \p child in ``named_children()``, else nullptr.
    module::Module *find_parent_module(module::Module *child) const;

    std::vector<module::Module *> module_live_;

    mutable bool module_parameter_cache_dirty_ = true;

    mutable std::vector<std::pair<std::string, TensorNode *>>
        module_parameter_cache_;

    bool grad_enabled_ = true;
    bool auto_tensor_name_phase_suffix_ = false;
    Index auto_phase_suffix_seq_ = 0;
    std::string tensor_name_suffix_tag_;
    std::vector<TensorPhaseArchiveEntry> tensor_phase_archives_;
    std::unique_ptr<NNGraphExecState> exec_;
    std::optional<FinishedTensorPhase> pending_compile_phase_;
    Index finished_phase_serial_ = 0;
};

//! True if any input requires grad.
bool any_input_requires_grad(const std::vector<NNGraph::TensorNode *> &inputs);

} // namespace nntile::graph
