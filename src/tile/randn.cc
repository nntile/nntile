/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/randn.cc
 * Randn operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/randn.hh"

namespace nntile
{

template<typename T>
void randn_async(const Tile<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean, T stddev)
{
    // Check inputs
    if(dst.ndim != offset.size())
    {
        throw std::runtime_error("dst.ndim != offset.size()");
    }
    if(dst.ndim != shape.size())
    {
        throw std::runtime_error("dst.ndim != shape.size()");
    }
    if(dst.ndim != stride.size())
    {
        throw std::runtime_error("dst.ndim != stride.size()");
    }
    // Treat special case of ndim=0
    if(dst.ndim == 0)
    {
        starpu_task_insert(randn_ndim0_get_codelet<T>(),
                STARPU_VALUE, &seed, sizeof(seed),
                STARPU_VALUE, &mean, sizeof(mean),
                STARPU_VALUE, &stddev, sizeof(stddev),
                STARPU_W, static_cast<starpu_data_handle_t>(dst),
                // 2 flops per single element
                STARPU_FLOPS, static_cast<fp64_t>(2),
                0);
        return;
    }
    // Treat non-zero ndim
    if(offset[0] < 0)
    {
        throw std::runtime_error("offset[0] < 0");
    }
    if(offset[0]+dst.shape[0] > shape[0])
    {
        throw std::runtime_error("offset[0]+dst.shape[0] > shape[0]");
    }
    if(stride[0] != 1)
    {
        throw std::runtime_error("stride[0] != 1");
    }
    Index jump = offset[0]; // stride[0] = 1
    Index prod_shape = 1;
    for(Index i = 1; i < dst.ndim; ++i)
    {
        if(offset[i] < 0)
        {
            throw std::runtime_error("offset[i] < 0");
        }
        if(offset[i]+dst.shape[i] > shape[i])
        {
            throw std::runtime_error("offset[i]+dst.shape[i] > shape[i]");
        }
        prod_shape *= shape[i-1];
        if(stride[i] != prod_shape)
        {
            throw std::runtime_error("stride[i] != prod_shape");
        }
        jump += offset[i] * stride[i];
    }
    seed = CORE_rnd64_jump(jump, seed);
    starpu_task_insert(randn_get_codelet<T>(),
            STARPU_VALUE, &(dst.ndim), sizeof(dst.ndim),
            STARPU_VALUE, &(dst.nelems), sizeof(dst.nelems),
            STARPU_VALUE, &seed, sizeof(seed),
            STARPU_VALUE, &mean, sizeof(mean),
            STARPU_VALUE, &stddev, sizeof(stddev),
            STARPU_VALUE, &(dst.shape[0]), dst.ndim*sizeof(dst.shape[0]),
            STARPU_VALUE, &(dst.stride[0]), dst.ndim*sizeof(dst.stride[0]),
            STARPU_VALUE, &(stride[0]), dst.ndim*sizeof(stride[0]),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            // 2 flops per single element
            STARPU_FLOPS, static_cast<fp64_t>(2*dst.nelems),
            0);
}

template
void randn_async(const Tile<fp32_t> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, fp32_t mean, fp32_t stddev);

template
void randn_async(const Tile<fp64_t> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, fp64_t mean, fp64_t stddev);

} // namespace nntile

