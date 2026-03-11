/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/module/rms_norm.hh
 * RMSNorm module - wraps gamma parameter and rms_norm graph op.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/io/safetensors.hh>
#include <nntile/graph/module/module.hh>

namespace nntile::graph::module
{

//! RMSNorm module: y = gamma * (x / sqrt(mean(x^2) + eps))
class RMSNorm : public Module
{
private:
    graph::NNGraph::TensorNode* gamma_tensor_ = nullptr;
    Index normalized_shape_;
    Index axis_;
    float eps_;
    int redux_;
    graph::DataType dtype_;

public:
    //! Constructor
    RMSNorm(graph::NNGraph* graph,
            const std::string& name,
            Index normalized_shape,
            Index axis = 0,
            float eps = 1e-6f,
            int redux = 0,
            graph::DataType dtype = graph::DataType::FP32);

    //! Forward pass
    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* x);

    //! Get string representation
    std::string repr() const override;

    void import_hf(const graph::io::SafeTensorsReader& reader,
                   const std::string& hf_prefix) override;

    void export_hf(graph::io::SafeTensorsWriter& writer,
                   const std::string& hf_prefix) const override;

    graph::NNGraph::TensorNode* gamma_tensor() const { return gamma_tensor_; }
};

} // namespace nntile::graph::module
