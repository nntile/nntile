/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/concat.cc
 * TensorGraph concat operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/concat.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/copy_intersection.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_concat(TensorGraph::Runtime& runtime,
                TensorGraph::TensorNode* a,
                TensorGraph::TensorNode* b,
                TensorGraph::TensorNode* output,
                Index axis)
{
    auto& a_t = runtime.get_tensor<T>(a);
    auto& b_t = runtime.get_tensor<T>(b);
    auto& out_t = runtime.get_tensor<T>(output);

    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    Index n_dim = static_cast<Index>(a_shape.size());

    if(axis < 0 || axis >= n_dim)
    {
        throw std::invalid_argument(
            "concat: axis out of range");
    }

    // Clear output before filling: copy_intersection uses STARPU_RW for edge
    // tiles when doing partial copies; uninitialized handles cause assert.
    nntile::tensor::clear<T>(out_t);

    // Copy a to output[0:a.shape[axis], ...]
    std::vector<Index> src_offset(n_dim, 0);
    std::vector<Index> dst_offset(n_dim, 0);
    nntile::tensor::copy_intersection<T>(a_t, src_offset, out_t, dst_offset);

    // Copy b to output[a.shape[axis]:, ...]
    dst_offset[axis] = a_shape[axis];
    nntile::tensor::copy_intersection<T>(b_t, src_offset, out_t, dst_offset);
}

} // namespace

TensorGraph::TensorNode* concat(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    Index axis,
    const std::string& output_name)
{
    if(a == nullptr || b == nullptr)
        throw std::invalid_argument("concat: input tensors must be non-null");
    if(a->graph() != b->graph())
        throw std::invalid_argument(
            "concat: tensors must belong to same graph");
    if(a->dtype() != b->dtype())
        throw std::invalid_argument(
            "concat: tensors must have same dtype");
    if(a->ndim() != b->ndim())
        throw std::invalid_argument(
            "concat: tensors must have same number of dimensions");

    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    Index n_dim = static_cast<Index>(a_shape.size());

    if(axis < 0 || axis >= n_dim)
        throw std::invalid_argument("concat: axis out of range");

    // Output shape: same as a and b except on axis
    std::vector<Index> output_shape = a_shape;
    output_shape[axis] = a_shape[axis] + b_shape[axis];
    for(Index i = 0; i < n_dim; ++i)
    {
        if(i != axis && a_shape[i] != b_shape[i])
        {
            throw std::invalid_argument(
                "concat: non-concat dimensions must match");
        }
    }

    TensorGraph* graph = a->graph();
    TensorGraph::TensorNode* output = graph->data(
        output_shape, output_name, a->dtype());

    auto op = std::make_shared<TensorConcatOp>(a, b, output, axis);
    graph->add_op(op);

    return output;
}

void TensorConcatOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(a);
    switch(dtype)
    {
        case DataType::FP32:
            run_concat<nntile::fp32_t>(runtime, a, b, output, axis);
            break;
        case DataType::FP32_FAST_TF32:
            run_concat<nntile::fp32_fast_tf32_t>
                (runtime, a, b, output, axis);
            break;
        case DataType::FP32_FAST_FP16:
            run_concat<nntile::fp32_fast_fp16_t>
                (runtime, a, b, output, axis);
            break;
        case DataType::FP32_FAST_BF16:
            run_concat<nntile::fp32_fast_bf16_t>
                (runtime, a, b, output, axis);
            break;
        case DataType::FP64:
            run_concat<nntile::fp64_t>(runtime, a, b, output, axis);
            break;
        case DataType::FP16:
            run_concat<nntile::fp16_t>(runtime, a, b, output, axis);
            break;
        case DataType::BF16:
            run_concat<nntile::bf16_t>(runtime, a, b, output, axis);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for concat");
        default:
            throw std::runtime_error("Unsupported data type for concat");
    }
}

} // namespace nntile::graph::tensor
