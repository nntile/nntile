/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/rms_norm.hh
 * RMSNorm module - wraps gamma parameter and rms_norm graph op.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! RMSNorm module: y = gamma * (x / sqrt(mean(x^2) + eps))
class RMSNorm : public Module
{
private:
    NNGraph::TensorNode* gamma_tensor_ = nullptr;
    Index normalized_shape_;
    Index axis_;
    float eps_;
    int redux_;
    DataType dtype_;

public:
    //! Constructor
    RMSNorm(NNGraph* graph,
            const std::string& name,
            Index normalized_shape,
            Index axis = -1,
            float eps = 1e-6f,
            int redux = 0,
            DataType dtype = DataType::FP32);

    //! Forward pass
    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* x);

    //! Get string representation
    std::string repr() const override;

    NNGraph::TensorNode* gamma_tensor() const { return gamma_tensor_; }
};

} // namespace nntile::module
