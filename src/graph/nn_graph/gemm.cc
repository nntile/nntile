/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/nn_graph/gemm.cc
 * NNGraph GEMM autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/gemm.hh"
#include "nntile/graph/logical/gemm.hh"

#include <stdexcept>

namespace nntile::graph
{

NNGraph::TensorNode* Gemm::build_forward(
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
        throw std::invalid_argument(
            "Gemm::build_forward: a and b must be non-null");
    }
    NNGraph& graph = a->graph();
    LogicalGraph::TensorNode& c_data = gemm(
        a->data(), b->data(), output_name, alpha, trans_a, trans_b, ndim,
        batch_ndim);
    bool out_requires_grad = any_input_requires_grad({a, b});
    NNGraph::TensorNode* c = graph.tensor(c_data, out_requires_grad);
    register_op(graph, {a, b}, c, GemmAttrs{trans_a, trans_b, alpha, 0.0, ndim, batch_ndim},
                [](const NNGraph::OpNode* op) { Gemm::build_backward(op); }, {});
    return c;
}

void Gemm::build_backward(const NNGraph::OpNode* op)
{
    NNGraph& graph = op->output()->graph();
    NNGraph::TensorNode* grad_out = op->output()->grad();
    const auto& attrs = std::get<GemmAttrs>(op->attrs());
    Scalar alpha = attrs.alpha;
    bool trans_a = attrs.trans_a;
    bool trans_b = attrs.trans_b;
    Index ndim = attrs.ndim;
    Index batch_ndim = attrs.batch_ndim;
    const auto& inputs = op->inputs();
    if(inputs.size() < 2 || grad_out == nullptr)
    {
        return;
    }
    NNGraph::TensorNode* a_nn = inputs[0];
    NNGraph::TensorNode* b_nn = inputs[1];

    // grad_A: depends on trans_a, trans_b from forward
    // trans_a=F,trans_b=F: grad_A = alpha*grad_C@B^T  -> gemm(grad_C,B, false,true)
    // trans_a=F,trans_b=T: grad_A = alpha*grad_C@B    -> gemm(grad_C,B, false,false)
    // trans_a=T,trans_b=F: grad_A = alpha*B@grad_C^T  -> gemm(B,grad_C, false,true)
    // trans_a=T,trans_b=T: grad_A = alpha*B^T@grad_C^T -> gemm(B,grad_C, true,true)
    if(a_nn != nullptr && a_nn->requires_grad())
    {
        bool first = graph.is_first_grad(a_nn);
        NNGraph::TensorNode* grad_a =
            graph.get_or_create_grad(a_nn, a_nn->name() + "_grad");
        Scalar beta = first ? 0.0 : 1.0;
        if(!trans_a)
        {
            gemm(grad_out->data(), b_nn->data(), grad_a->data(), alpha, beta,
                 false, !trans_b, ndim, batch_ndim);
        }
        else
        {
            gemm(b_nn->data(), grad_out->data(), grad_a->data(), alpha, beta,
                 trans_b, true, ndim, batch_ndim);
        }
    }

    // grad_B: depends on trans_a, trans_b from forward
    // trans_a=F,trans_b=F: grad_B = alpha*A^T@grad_C   -> gemm(A,grad_C, true,false)
    // trans_a=F,trans_b=T: grad_B = alpha*grad_C^T@A    -> gemm(grad_C,A, true,false)
    // trans_a=T,trans_b=F: grad_B = alpha*A@grad_C     -> gemm(A,grad_C, false,false)
    // trans_a=T,trans_b=T: grad_B = alpha*grad_C^T@A^T -> gemm(grad_C,A, true,true)
    if(b_nn != nullptr && b_nn->requires_grad())
    {
        bool first = graph.is_first_grad(b_nn);
        NNGraph::TensorNode* grad_b =
            graph.get_or_create_grad(b_nn, b_nn->name() + "_grad");
        Scalar beta = first ? 0.0 : 1.0;
        if(!trans_b)
        {
            gemm(a_nn->data(), grad_out->data(), grad_b->data(), alpha, beta,
                 !trans_a, false, ndim, batch_ndim);
        }
        else
        {
            gemm(grad_out->data(), a_nn->data(), grad_b->data(), alpha, beta,
                 true, trans_a, ndim, batch_ndim);
        }
    }
}

} // namespace nntile::graph
