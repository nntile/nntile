/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/t5/t5_ff.hh
 * T5LayerFF - layer_norm + gated MLP (GELUTANH / HF gelu_new) + residual.
 *
 * T5 uses: layer_norm -> DenseReluDense -> add(x, ...)
 * DenseReluDense: wo(gate(wi_0(x)) * wi_1(x)) with GELU activation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// NNTile headers
#include <nntile/graph.hh>
#include <nntile/model/t5/t5_config.hh>
#include <nntile/module/activation.hh>
#include <nntile/module/gated_mlp.hh>
#include <nntile/module/rms_norm.hh>

namespace nntile::model::t5
{

//! T5LayerFF - layer_norm -> gated MLP (GELUTANH) -> residual add
//! Architecture: x + wo(GELUTANH(wi_0(x)) * wi_1(x))
class T5LayerFF : public module::Module
{
private:
    module::RMSNorm layer_norm_;
    module::GatedMlp dense_;  // GELU gated MLP

    T5Config config_;
    DataType dtype_;

public:
    T5LayerFF(NNGraph* graph,
              const std::string& name,
              const T5Config& config,
              DataType dtype = DataType::FP32);

    NNGraph::TensorNode* forward(
        NNGraph::TensorNode* input);

    std::string repr() const override;
};

} // namespace nntile::model::t5
