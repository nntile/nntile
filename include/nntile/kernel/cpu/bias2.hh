/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/bias2.hh
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include <nntile/base_types.hh>

namespace nntile
{

template<typename T>
void bias2_kernel_cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept;

template<typename T>
void bias2_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

