/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/fp16_to_fp32.cc
 * Convert fp32_t array into fp16_t array
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-04
 * */

#include "nntile/tile/fp16_to_fp32.hh"
#include "nntile/starpu/fp16_to_fp32.hh"

namespace nntile
{
namespace tile
{

void fp16_to_fp32_async(const Tile<fp16_t> &src, const Tile<fp32_t> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit forward relu
    starpu::fp16_to_fp32::submit(src.nelems, src, dst);
}

void fp16_to_fp32(const Tile<fp16_t> &src, const Tile<fp32_t> &dst)
{
    fp16_to_fp32_async(src, dst);
    starpu_task_wait_for_all();
}

} // namespace tile
} // namespace nntile
