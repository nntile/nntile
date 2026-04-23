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

namespace nntile::graph::tensor
{



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

} // namespace nntile::graph::tensor
