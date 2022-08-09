/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/normalize.cc
 * Normalize operation for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

namespace nntile
{
namespace starpu

//! Renormalize buffer along middle axis of StarPU buffer
template<typename T>
void normalize_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<normalize_starpu_args<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    // Launch kernel
    const T *gamma_beta = interfaces[0]->get_ptr<T>();
    T gamma = gamma_beta[0], beta = gamma_beta[1];
    const T *sumnorm = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    normalize_kernel_cpu<T>(args->m, args->n, args->k, args->l, args->eps,
            gamma, beta, sumnorm, dst);
}

// Explicit instantiation of templates
template
void normalize_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void normalize_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace starpu
} // namespace nntile

