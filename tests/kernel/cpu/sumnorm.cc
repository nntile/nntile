/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/sumnorm.cc
 * Sum and Euclidian norm of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-02
 * */

#include "nntile/kernel/cpu/sumnorm.hh"
#include "nntile/kernel/args/sumnorm.hh"
#include "nntile/starpu.hh"
#include <vector>
#include <stdexcept>
#include <limits>

using namespace nntile;

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    // Init test input
    std::vector<T> src(m*n*k), sumnorm(2*m*n);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                src[(i1*k+i2)*m+i0] = T(i0+i1+i2) / T{10};
            }
        }
    }
    // Check low-level kernel
    sumnorm_kernel_cpu<T>(m, n, k, &src[0], &sumnorm[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T sum_ref = k * (2*a+k-1) / 2 / T{10};
            T sum = sumnorm[2*(i1*m+i0)];
            if(std::abs(sum/sum_ref-T{1}) / std::numeric_limits<T>::epsilon()
                    > 10)
            {
                throw std::runtime_error("Wrong sum");
            }
            T norm_sqr_ref = k * (2*(a-1)*a+(2*a+k-1)*(2*a+2*k-1)) / 6
                / T{100};
            T norm = sumnorm[2*(i1*m+i0)+1];
            T norm_sqr = norm * norm;
            if(std::abs(norm_sqr/norm_sqr_ref-T{1})
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong norm");
            }
        }
    }
    // Check low-level kernel even more
    std::vector<T> sumnorm2(sumnorm);
    sumnorm_kernel_cpu<T>(m, n, k, &src[0], &sumnorm2[0]);
    sumnorm_kernel_cpu<T>(m, n, k, &src[0], &sumnorm2[0]);
    sumnorm_kernel_cpu<T>(m, n, k, &src[0], &sumnorm2[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = 2 * (i1*m+i0);
            if(std::abs(sumnorm2[i]/sumnorm[i]-T{4})
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong quadruple sum");
            }
            if(std::abs(sumnorm2[i+1]/sumnorm[i+1]-T{2})
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong double norm");
            }
        }
    }
    // Now check StarPU codelet
    // StarPU interfaces
    StarpuVariableInterface src_interface(&src[0], m*n*k*sizeof(T)),
            sumnorm_interface(&sumnorm[0], 2*m*n*sizeof(T));
    // Codelet arguments
    sumnorm_starpu_args args =
    {
        .m = m,
        .n = n,
        .k = k
    };
    void *buffers[2] = {&src_interface, &sumnorm_interface};
    // Launch codelet
    sumnorm_starpu_cpu<T>(buffers, &args);
    sumnorm_starpu_cpu<T>(buffers, &args);
    sumnorm_starpu_cpu<T>(buffers, &args);
    // Check it
    for(Index i = 0; i < 2*m*n; ++i)
    {
        if(sumnorm[i] != sumnorm2[i])
        {
            throw std::runtime_error("Starpu codelet wrong result");
        }
    }
}

int main(int argc, char **argv)
{
    validate<fp32_t>(1, 9, 10);
    validate<fp32_t>(8, 9, 1);
    validate<fp32_t>(8, 1, 10);
    validate<fp32_t>(4, 7, 8);
    validate<fp64_t>(1, 9, 10);
    validate<fp64_t>(8, 9, 1);
    validate<fp64_t>(8, 1, 10);
    validate<fp64_t>(4, 7, 8);
    return 0;
}

