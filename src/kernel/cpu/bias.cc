/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/bias.cc
 * Bias operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-02
 * */

#include "nntile/kernel/cpu/bias.hh"
#include "nntile/kernel/args/bias.hh"
#include "nntile/starpu.hh"

namespace nntile
{

//! Apply bias along middle axis
//
// For a provided m-by-k-by-n output tensor dst apply bias along second axis
// with k elements from m-by-n tensor src. A value src[i, j] is added to the
// entire slice dst[i, :, j].
//
// @param[in] m: Size of the first mode of src and dst tensors
// @param[in] n: Size of the last mode of src and dst tensors
// @param[in] k: Size of the middle mode of dst tensor
// @param[in] src: Source of the bias
// @param[inout] dst: Destination of the bias
//
// @sa bias_starpu_cpu
template<typename T>
void bias_kernel_cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept
{
    Index src_offset = 0;
    const Index mk = m * k;
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Output slice to be updated
            T *dst_slice = dst + i2*mk + i1;
            const T src_val = src[src_offset];
            ++src_offset;
            // Cycle over slice of output buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T &dst_val = dst_slice[i0*m];
                // And update it
                dst_val = dst_val + src_val;
            }
        }
    }
}

//! Apply bias along middle axis of StarPU buffer
//
// For a provided m-by-k-by-n output tensor dst apply bias along second axis
// with k elements from m-by-n tensor src. A value src[i, j] is added to the
// entire slice dst[i, :, j].
//
// @param[in] buffers: input src and output dst tensors through StarPU
//      handles
// @param[in] cl_args: Sizes m, n and k
//
// @sa bias_kernel_cpu
template<typename T>
void bias_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<bias_starpu_args *>(cl_args);
    // Get interfaces
    auto interface = reinterpret_cast<StarpuVariableInterface **>(buffers);
    // Launch kernel
    const T *src = interface[0]->get_ptr<T>();
    T *dst = interface[1]->get_ptr<T>();
    bias_kernel_cpu<T>(args->m, args->n, args->k, src, dst);
}

// Explicit instantiation
template
void bias_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void bias_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

