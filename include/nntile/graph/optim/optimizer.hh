/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/optim/optimizer.hh
 * Base Optimizer class for parameter optimization (like PyTorch's
 * torch.optim.Optimizer). Collects parameter-gradient pairs from a Module
 * and manages optimizer state tensors (e.g. velocity, moments).
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>
#include <vector>

#include <nntile/graph/nn.hh>
#include <nntile/graph/tensor.hh>
#include <nntile/graph/tile/graph_runtime.hh>
#include <nntile/graph/io/safetensors.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::optim
{

class Optimizer
{
protected:
    NNGraph* graph_;

    struct ParamState
    {
        std::string name;
        NNGraph::TensorNode* param;
        NNGraph::TensorNode* grad;
        std::vector<std::pair<std::string, NNGraph::TensorNode*>> buffers;
    };
    std::vector<ParamState> param_states_;

    Index num_iter_ = 0;

    void collect_params(module::Module* module);

public:
    Optimizer(NNGraph* graph, module::Module* module);
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    Optimizer(Optimizer&&) = delete;
    Optimizer& operator=(Optimizer&&) = delete;

    NNGraph* graph() { return graph_; }
    const NNGraph* graph() const { return graph_; }

    //! Add optimizer step operations to the graph.
    //! Must be called after backward() and before compile().
    virtual void step() = 0;

    //! All state tensor name/pointer pairs (for binding zero data).
    std::vector<std::pair<std::string, NNGraph::TensorNode*>>
        named_state_tensors() const;

    //! Save optimizer state tensors to a SafeTensors file.
    void save(const std::string& path) const;

    //! Load optimizer state tensors from a SafeTensors file.
    void load(const std::string& path);

    //! Save optimizer config (hyperparameters + num_iter) to a JSON file.
    virtual void save_config(const std::string& path) const = 0;

    //! Load optimizer config from a JSON file.
    virtual void load_config(const std::string& path) = 0;

    //! Import optimizer state from HuggingFace-style SafeTensors.
    void import_hf(const io::SafeTensorsReader& reader,
                   const std::string& prefix);

    //! Export optimizer state to HuggingFace-style SafeTensors.
    void export_hf(io::SafeTensorsWriter& writer,
                   const std::string& prefix) const;

    //! Sync optimizer state tensors from the runtime back to bind_hints
    //! so that save() / export_hf() can access the trained state.
    //! Must be called after runtime.wait() and before save().
    void sync_from_runtime(TileGraph::Runtime& runtime);

    Index num_iter() const { return num_iter_; }
    void set_num_iter(Index n) { num_iter_ = n; }

    virtual std::string repr() const;
    std::string to_string() const;
    void print() const;
};

} // namespace nntile::graph::optim
