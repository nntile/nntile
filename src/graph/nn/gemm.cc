/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/gemm.cc
 * NNGraph GEMM autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/gemm.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/gemm.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar gemm_new_output_beta = 0.0;
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNGemmOp::forward(const std::string& output_name)
{
    if(a == nullptr || b == nullptr)
    {
        throw std::invalid_argument(
            "NNGemmOp::forward: a, b must be non-null");
    }
    NNGraph* graph = a->graph();
    std::vector<Index> c_shape = gemm_output_shape(
        a->shape(), b->shape(), trans_a, trans_b, ndim, batch_ndim);
    bool out_requires_grad = any_input_requires_grad({a, b});
    NNGraph::TensorNode* c = graph->tensor(
        std::move(c_shape), output_name, a->dtype(), out_requires_grad);
    outputs_ = {c};
    graph::gemm(a->data(), b->data(), c->data(),
                alpha, gemm_new_output_beta, trans_a, trans_b, ndim, batch_ndim);
    return c;
}

void NNGemmOp::backward() const
{
    NNGraph* graph = a->graph();
    NNGraph::TensorNode* grad_out = output()->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(a != nullptr && a->requires_grad())
    {
        bool first = graph->is_first_grad(a);
        NNGraph::TensorNode* grad_a =
            graph->get_or_create_grad(a, a->name() + "_grad");
        Scalar beta = first ? grad_overwrite : grad_accumulate;
        if(!trans_a)
        {
            graph::gemm(grad_out->data(), b->data(), grad_a->data(),
                       alpha, beta, false, !trans_b, ndim, batch_ndim);
        }
        else
        {
            graph::gemm(b->data(), grad_out->data(), grad_a->data(),
                       alpha, beta, trans_b, true, ndim, batch_ndim);
        }
    }
    if(b != nullptr && b->requires_grad())
    {
        bool first = graph->is_first_grad(b);
        NNGraph::TensorNode* grad_b =
            graph->get_or_create_grad(b, b->name() + "_grad");
        Scalar beta = first ? grad_overwrite : grad_accumulate;
        if(!trans_b)
        {
            graph::gemm(a->data(), grad_out->data(), grad_b->data(),
                       alpha, beta, !trans_a, false, ndim, batch_ndim);
        }
        else
        {
            graph::gemm(grad_out->data(), a->data(), grad_b->data(),
                       alpha, beta, true, trans_a, ndim, batch_ndim);
        }
    }
}

NNGraph::TensorNode* gemm(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    if(a == nullptr || b == nullptr)
    {
        throw std::invalid_argument("gemm: a and b must be non-null");
    }
    NNGraph* graph = a->graph();
    auto op = std::make_shared<NNGemmOp>(
        a, b, alpha, trans_a, trans_b, ndim, batch_ndim);
    NNGraph::TensorNode* c = op->forward(output_name);
    register_op(*graph, std::move(op));
    return c;
}

} // namespace nntile::graph
