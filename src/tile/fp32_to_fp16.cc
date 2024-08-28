/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/fp32_to_fp16.cc
 * Convert fp32_t array into fp16_t array
 *
 * @version 1.1.0
 * */

#include "nntile/tile/fp32_to_fp16.hh"
#include "nntile/starpu/fp32_to_fp16.hh"

namespace nntile::tile
{

void fp32_to_fp16_async(const Tile<fp32_t> &src, const Tile<fp16_t> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit forward relu
    starpu::fp32_to_fp16::submit(src.nelems, src, dst);
}

void fp32_to_fp16(const Tile<fp32_t> &src, const Tile<fp16_t> &dst)
{
    fp32_to_fp16_async(src, dst);
    starpu_task_wait_for_all();
}

} // namespace nntile::tile
