/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/normalize.cc
 * Normalize operation for a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-02
 * */

#include "nntile/kernel/cpu/normalize.hh"
#include "nntile/kernel/args/normalize.hh"
#include "nntile/starpu.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>

using namespace nntile;

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k, Index l, T eps, T gamma, T beta)
{
    // Init test input
    std::vector<T> sumnorm(2*m*n), dst(m*n*k);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                dst[(i1*k+i2)*m+i0] = T(i0+i1+i2) / T{10};
            }
        }
    }
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            T avg = T(i0+i1) / T{10};
            sumnorm[2*(i1*m+i0)] = avg * T(l);
            sumnorm[2*(i1*m+i0)+1] = std::sqrt((avg*avg+T{1}+eps) * T(l));
        }
    }
    std::vector<T> dst2(dst);
    // Check low-level kernel
    normalize_kernel_cpu<T>(m, n, k, l, eps, gamma, beta, &sumnorm[0], &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = T(i2) / T{10} / std::sqrt(T{1}+2*eps) * gamma + beta;
                if(std::abs(val/val_ref-T{1})
                        / std::numeric_limits<T>::epsilon() > 50)
                {
                    throw std::runtime_error("Wrong value");
                }
            }
        }
    }
    // Now check StarPU codelet
    // StarPU interfaces
    T gamma_beta[2] = {gamma, beta};
    StarpuVariableInterface sumnorm_interface(&sumnorm[0], 2*m*n*sizeof(T)),
            dst2_interface(&dst2[0], m*n*k*sizeof(T)),
            gamma_beta_interface(gamma_beta, sizeof(gamma_beta));
    // Codelet arguments
    normalize_starpu_args<T> args =
    {
        .m = m,
        .n = n,
        .k = k,
        .l = l,
        .eps = eps
    };
    void *buffers[3] = {&gamma_beta_interface, &sumnorm_interface,
        &dst2_interface};
    // Launch codelet
    normalize_starpu_cpu<T>(buffers, &args);
    // Check it
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst2[i])
        {
            throw std::runtime_error("Starpu codelet wrong result");
        }
    }
}

int main(int argc, char **argv)
{
    fp64_t eps[3] = {0.0, 1.0, 11.1};
    fp64_t gamma[3] = {0.0, 1.0, 3.3};
    fp64_t beta[3] = {0.0, 1.1, -2.2};
    for(Index i = 0 ; i < sizeof(eps)/sizeof(eps[0]); ++i)
    {
        for(Index j = 0 ; j < sizeof(gamma)/sizeof(gamma[0]); ++j)
        {
            for(Index k = 0 ; k < sizeof(beta)/sizeof(beta[0]); ++k)
            {
                validate<fp32_t>(1, 9, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp32_t>(8, 1, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp32_t>(8, 9, 1, 22, eps[i], gamma[j], beta[k]);
                validate<fp64_t>(1, 9, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp64_t>(8, 1, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp64_t>(8, 9, 1, 22, eps[i], gamma[j], beta[k]);
            }
        }
    }
    return 0;
}

