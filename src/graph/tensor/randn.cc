/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/randn.cc
 * TensorGraph randn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/randn.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/randn.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_randn(
    TensorGraph::Runtime& runtime,
    const std::vector<Index>& start,
    const std::vector<Index>& underlying_shape,
    unsigned long long seed,
    Scalar mean, Scalar stddev,
    TensorGraph::TensorNode* dst)
{
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::randn<T>(dst_t, start, underlying_shape, seed, mean, stddev);
}

} // namespace

void randn(
    TensorGraph::TensorNode* dst,
    const std::vector<Index>& start,
    const std::vector<Index>& underlying_shape,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "randn: dst tensor must be non-null");
    }
    if(start.size() != underlying_shape.size())
    {
        throw std::invalid_argument(
            "randn: start and underlying_shape must have same size");
    }
    if(dst->ndim() != static_cast<Index>(start.size()))
    {
        throw std::invalid_argument(
            "randn: start size must match dst ndim");
    }

    auto op = std::make_shared<TensorRandnOp>(
        dst, start, underlying_shape, seed, mean, stddev);
    dst->graph()->add_op(op);
}

void TensorRandnOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_randn<nntile::fp32_t>(
                runtime, start, underlying_shape, seed, mean, stddev, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_randn<nntile::fp32_fast_tf32_t>(
                runtime, start, underlying_shape, seed, mean, stddev, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_randn<nntile::fp32_fast_fp16_t>(
                runtime, start, underlying_shape, seed, mean, stddev, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_randn<nntile::fp32_fast_bf16_t>(
                runtime, start, underlying_shape, seed, mean, stddev, dst);
            break;
        case DataType::FP64:
            run_randn<nntile::fp64_t>(
                runtime, start, underlying_shape, seed, mean, stddev, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 data type not supported for randn operation");
        case DataType::BF16:
            run_randn<nntile::bf16_t>(
                runtime, start, underlying_shape, seed, mean, stddev, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for randn operation");
        default:
            throw std::runtime_error("Unsupported data type for randn");
    }
}

} // namespace nntile::graph
