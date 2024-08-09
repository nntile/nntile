/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/subtract_indexed_outputs.cc
 *
 * @version 1.1.0
 * */

#include "nntile/tile/subtract_indexed_outputs.hh"
#include "nntile/starpu/subtract_indexed_outputs.hh"

namespace nntile::tile
{

template<typename T>
void subtract_indexed_outputs_async(Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst)
{
// TODO - add description
    if(labels.ndim != dst.ndim-1)
    {
        throw std::runtime_error("labels.ndim != dst.ndim-1");
    }
    for(Index i = 0; i < labels.ndim; ++i)
    {
        if(labels.shape[i] != dst.shape[i+1])
        {
            throw std::runtime_error("labels.shape[i] != dst.shape[i+1]");
        }
    }
    // Insert task
    starpu::subtract_indexed_outputs::submit<T>(dst.shape[0], labels.nelems,
            val, labels, dst);
}

template<typename T>
void subtract_indexed_outputs(Scalar val, const Tile<int64_t> &labels,
        const Tile<T> &dst)
{
    subtract_indexed_outputs_async<T>(val, labels, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void subtract_indexed_outputs_async<fp32_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_t> &dst);

template
void subtract_indexed_outputs_async<fp32_fast_tf32_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp32_fast_tf32_t> &dst);

template
void subtract_indexed_outputs_async<fp64_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<fp64_t> &dst);

template
void subtract_indexed_outputs_async<bf16_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst);

// Explicit instantiation
template
void subtract_indexed_outputs<fp32_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_t> &dst);

template
void subtract_indexed_outputs<fp32_fast_tf32_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp32_fast_tf32_t> &dst);

template
void subtract_indexed_outputs<fp64_t>(Scalar val, const Tile<int64_t> &labels,
        const Tile<fp64_t> &dst);

template
void subtract_indexed_outputs<bf16_t>(Scalar val,
        const Tile<int64_t> &labels, const Tile<bf16_t> &dst);

} // namespace nntile::tile
