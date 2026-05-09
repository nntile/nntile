/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/graph.cc
 * Implementation of NNGraph class (include/nntile/graph/nn/graph.hh).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/graph.hh"

#include "nntile/graph/module/module.hh"
#include "nntile/graph/nn/graph_data_node.hh"
#include "nntile/graph/nn/graph_op_node.hh"
#include "nntile/graph/tile/append_tensor_graph_phase.hh"

#include <algorithm>
#include <nntile/graph/tile/graph_runtime.hh>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nntile::graph
{

struct NNGraphExecState
{
    std::optional<TileGraph> tile_graph;
    TileGraphIncrementalState inc_state;
    TensorNodeToTileMap tile_map;
    std::optional<TileGraphExecutor> runtime;
};

NNGraph::~NNGraph() = default;

NNGraph::NNGraph(const std::string &name) : name_(name), tensor_graph_(name) {}

void NNGraph::clear_pending_compile_if_same(
    FinishedTensorPhase const &compiled)
{
    if (!pending_compile_phase_.has_value())
    {
        return;
    }
    FinishedTensorPhase const &pend = *pending_compile_phase_;
    if (pend.tensor_graph != compiled.tensor_graph)
    {
        return;
    }
    if (pend.snapshot.op_begin != compiled.snapshot.op_begin ||
        pend.snapshot.op_end != compiled.snapshot.op_end)
    {
        return;
    }
    if (pend.snapshot.carried_tensors.size() !=
        compiled.snapshot.carried_tensors.size())
    {
        return;
    }
    for (size_t i = 0; i < pend.snapshot.carried_tensors.size(); ++i)
    {
        if (pend.snapshot.carried_tensors[i] !=
            compiled.snapshot.carried_tensors[i])
        {
            return;
        }
    }
    pending_compile_phase_.reset();
}

void NNGraph::ensure_exec_state()
{
    if (!exec_)
    {
        exec_ = std::make_unique<NNGraphExecState>();
    }
    if (!exec_->tile_graph.has_value())
    {
        exec_->tile_graph.emplace(name_ + "_tile");
    }
    if (!exec_->runtime.has_value())
    {
        exec_->runtime.emplace(*exec_->tile_graph);
    }
}

void NNGraph::lower_and_compile(TensorGraphTiling const &tiling)
{
    if (!pending_compile_phase_.has_value())
    {
        throw std::runtime_error(
            "NNGraph::lower_and_compile: call finish_phase() first");
    }
    ensure_exec_state();
    compile_incremental_nn_phase(*pending_compile_phase_,
        *this,
        tiling,
        *exec_->tile_graph,
        *exec_->runtime,
        exec_->inc_state,
        exec_->tile_map,
        true);
}

void NNGraph::lower_and_compile()
{
    lower_and_compile(TensorGraphTiling::from_tensor_graph(tensor_graph_));
}

TileGraph::Runtime &NNGraph::runtime()
{
    if (!exec_ || !exec_->runtime.has_value())
    {
        throw std::runtime_error(
            "NNGraph::runtime: call lower_and_compile() first");
    }
    return *exec_->runtime;
}

bool NNGraph::has_runtime() const
{
    return exec_ && exec_->runtime.has_value();
}

void NNGraph::set_tensor_name_suffix_tag(std::string tag)
{
    auto_tensor_name_phase_suffix_ = false;
    tensor_name_suffix_tag_ = std::move(tag);
}

void NNGraph::clear_tensor_name_suffix_tag()
{
    auto_tensor_name_phase_suffix_ = false;
    tensor_name_suffix_tag_.clear();
}

void NNGraph::enable_auto_tensor_name_phase_suffix(bool enable)
{
    if (enable)
    {
        auto_tensor_name_phase_suffix_ = true;
        auto_phase_suffix_seq_ = 0;
        tensor_name_suffix_tag_ = "0";
    }
    else
    {
        auto_tensor_name_phase_suffix_ = false;
    }
}

void NNGraph::bump_auto_tensor_name_phase_suffix_after_compile()
{
    if (!auto_tensor_name_phase_suffix_)
    {
        return;
    }
    ++auto_phase_suffix_seq_;
    tensor_name_suffix_tag_ =
        std::to_string(static_cast<long long>(auto_phase_suffix_seq_));
}

void NNGraph::push_tensor_phase_archive(TensorPhaseArchiveEntry entry)
{
    tensor_phase_archives_.push_back(std::move(entry));
}

void NNGraph::clear_tensor_phase_archives() { tensor_phase_archives_.clear(); }

TensorGraph::PhaseSnapshot NNGraph::seal_phase()
{
    return tensor_graph_.seal_phase();
}

TensorGraph::PhaseSnapshot NNGraph::seal_phase(
    std::vector<TensorNode const *> const &carried)
{
    std::vector<TensorGraph::TensorNode const *> data_carried;
    data_carried.reserve(carried.size());
    for (TensorNode const *n : carried)
    {
        if (n == nullptr)
        {
            throw std::invalid_argument(
                "NNGraph::seal_phase: carried tensor must be non-null");
        }
        data_carried.push_back(n->data());
    }
    return tensor_graph_.seal_phase(std::move(data_carried));
}

FinishedTensorPhase NNGraph::finish_phase(bool reset_autograd_state)
{
    if (pending_compile_phase_.has_value())
    {
        throw std::runtime_error(
            "NNGraph::finish_phase: pending slice not compiled; call "
            "lower_and_compile() before finishing another phase");
    }
    TensorGraph::PhaseSnapshot snap = seal_phase();
    ++finished_phase_serial_;
    pending_compile_phase_.emplace(
        FinishedTensorPhase{&tensor_graph_, std::move(snap)});
    if (reset_autograd_state)
    {
        clear_op_nodes();
        clear_producers_on_tensors();
    }
    return *pending_compile_phase_;
}

void NNGraph::reset_phase_seal_cursor()
{
    tensor_graph_.reset_phase_seal_cursor();
}

NNGraph::TensorNode *NNGraph::tensor(
    std::vector<Index> shape, DataType dtype, bool requires_grad)
{
    TensorGraph::TensorNode *data =
        tensor_graph_.data(std::move(shape), dtype);
    auto node = std::make_unique<TensorNode>(this, data, requires_grad);
    TensorNode *node_ptr = node.get();

    tensors_.push_back(std::move(node));

    return node_ptr;
}

NNGraph::TensorNode *NNGraph::tensor(
    TensorGraph::TensorNode *data, bool requires_grad)
{
    if (data == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::tensor: data tensor must be non-null");
    }
    if (data->graph() != &tensor_graph_)
    {
        throw std::invalid_argument("NNGraph::tensor: tensor must belong to "
                                    "this graph's tensor graph");
    }
    auto node = std::make_unique<TensorNode>(this, data, requires_grad);
    TensorNode *node_ptr = node.get();
    tensors_.push_back(std::move(node));
    return node_ptr;
}

NNGraph::TensorNode *NNGraph::get_tensor(TensorGraph::TensorNode const *data)
{
    if (data == nullptr)
    {
        return nullptr;
    }
    for (auto const &up : tensors_)
    {
        if (up->data() == data)
        {
            return up.get();
        }
    }
    return nullptr;
}

const NNGraph::TensorNode *NNGraph::get_tensor(
    TensorGraph::TensorNode const *data) const
{
    if (data == nullptr)
    {
        return nullptr;
    }
    for (auto const &up : tensors_)
    {
        if (up->data() == data)
        {
            return up.get();
        }
    }
    return nullptr;
}

std::vector<std::string> NNGraph::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (auto const &up : tensors_)
    {
        names.push_back(up->name());
    }
    return names;
}

void NNGraph::register_live_module(module::Module *mod)
{
    module_live_.push_back(mod);
    mark_module_parameter_cache_dirty();
}

void NNGraph::unregister_live_module(module::Module *mod)
{
    auto it = std::find(module_live_.begin(), module_live_.end(), mod);
    if (it != module_live_.end())
    {
        module_live_.erase(it);
    }
    mark_module_parameter_cache_dirty();
}

void NNGraph::mark_module_parameter_cache_dirty()
{
    module_parameter_cache_dirty_ = true;
}

void NNGraph::ensure_module_parameter_cache() const
{
    if (module_parameter_cache_dirty_)
    {
        rebuild_module_parameter_cache();
    }
}

void NNGraph::rebuild_module_parameter_cache() const
{
    module_parameter_cache_.clear();
    std::vector<module::Module *> roots;
    roots.reserve(8);
    for (module::Module *mod : module_live_)
    {
        if (mod->parent_ == nullptr)
        {
            roots.push_back(mod);
        }
    }
    if (roots.empty())
    {
        module_parameter_cache_dirty_ = false;
        return;
    }
    for (module::Module *root : roots)
    {
        root->append_parameter_tree_for_lazy_graph(module_parameter_cache_);
    }
    module_parameter_cache_dirty_ = false;
}

module::Module *NNGraph::find_parent_module(module::Module *child) const
{
    if (child == nullptr)
    {
        return nullptr;
    }
    for (module::Module *mod : module_live_)
    {
        for (const auto &entry : mod->named_children())
        {
            if (entry.second == child)
            {
                return mod;
            }
        }
    }
    return nullptr;
}

std::vector<NNGraph::TensorNode *> NNGraph::parameters() const
{
    ensure_module_parameter_cache();
    std::vector<TensorNode *> result;
    result.reserve(module_parameter_cache_.size());
    for (const auto &entry : module_parameter_cache_)
    {
        result.push_back(entry.second);
    }
    return result;
}

std::vector<std::pair<std::string, NNGraph::TensorNode *>>
NNGraph::named_parameters() const
{
    ensure_module_parameter_cache();
    return module_parameter_cache_;
}

bool NNGraph::requires_grad(const TensorNode *tensor) const
{
    return tensor != nullptr &&
           (tensor->requires_grad() || tensor->grad() != nullptr);
}

void NNGraph::set_requires_grad(TensorNode *tensor, bool value)
{
    if (tensor != nullptr)
    {
        tensor->set_requires_grad(value);
    }
}

std::pair<NNGraph::TensorNode *, bool> NNGraph::get_or_create_grad(
    TensorNode *tensor, const std::string &grad_name)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::get_or_create_grad: tensor is nullptr");
    }
    if (tensor->grad() != nullptr)
    {
        if (tensor->grad()->name() != grad_name)
        {
            throw std::invalid_argument(
                "NNGraph::get_or_create_grad: tensor '" + tensor->name() +
                "' already has gradient '" + tensor->grad()->name() +
                "' but caller requested '" + grad_name + "'");
        }
        return {tensor->grad(), false};
    }

    TensorGraph::TensorNode *grad_data =
        tensor_graph_.data(tensor->shape(), tensor->dtype());
    grad_data->set_name(grad_name);
    // Grad axes must match the tensor's axes (same tiling, same dimension
    // groups)
    grad_data->set_axes(tensor->data()->axes());
    auto grad_node = std::make_unique<TensorNode>(this, grad_data, false);
    TensorNode *grad_ptr = grad_node.get();
    tensors_.push_back(std::move(grad_node));
    tensor->set_grad(grad_ptr);
    tensor->set_requires_grad(true);
    return {grad_ptr, true};
}

void NNGraph::clear_op_nodes() { op_nodes_.clear(); }

void NNGraph::clear_producers_on_tensors()
{
    for (auto &t : tensors_)
    {
        if (t->has_producer())
        {
            t->set_producer(nullptr);
        }
    }
}

NNGraph::NoGradGuard::NoGradGuard(NNGraph *graph) :
    graph_(graph), prev_(graph != nullptr ? graph->grad_enabled_ : true)
{
    if (graph_ != nullptr)
    {
        graph_->grad_enabled_ = false;
    }
}

NNGraph::NoGradGuard::~NoGradGuard()
{
    if (graph_ != nullptr)
    {
        graph_->grad_enabled_ = prev_;
    }
}

std::string NNGraph::to_string() const
{
    std::stringstream ss;
    ss << "NNGraph(name='" << name_ << "', tensors=" << num_tensors()
       << ", autograd_ops=" << num_ops() << ")\n";

    ss << "Tensors:\n";
    for (const auto &t : tensors_)
    {
        ss << "  " << t->to_string() << "\n";
    }

    return ss.str();
}

void NNGraph::register_op(std::shared_ptr<OpNode> op)
{
    const bool need_backward =
        is_grad_enabled() && any_input_requires_grad(op->inputs());

    if (!need_backward)
    {
        return;
    }

    OpNode *op_nn = op.get();
    op_nodes_.push_back(std::move(op));

    for (TensorNode *out : op_nn->outputs())
    {
        if (out != nullptr)
        {
            out->set_producer(op_nn);
        }
    }
}

bool any_input_requires_grad(const std::vector<NNGraph::TensorNode *> &inputs)
{
    for (const auto *in : inputs)
    {
        if (in != nullptr && in->requires_grad())
        {
            return true;
        }
    }
    return false;
}

} // namespace nntile::graph
