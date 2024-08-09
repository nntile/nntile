/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/addcdiv/cuda.hh
 * Addcdiv operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::addcdiv
{

template<typename T>
void cuda(cudaStream_t stream, Scalar val, Scalar eps, Index nelems,
        const T *nom, const T *denom, T *res)
    noexcept;

} // namespace nntile::kernel::addcdiv
