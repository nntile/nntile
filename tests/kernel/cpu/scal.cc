/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/scal.cc
 * Scal operartion on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-03
 * */

#include "nntile/kernel/cpu/scal.hh"
#include "nntile/kernel/args/scal.hh"
#include "nntile/starpu.hh"
#include <vector>
#include <stdexcept>

using namespace nntile;

// Templated validation
template<typename T>
void validate(Index nelems, T alpha)
{
    // Init test input
    std::vector<T> dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = T(2*i+1-nelems) / T{10};
    }
    std::vector<T> dst2(dst);
    // Check low-level kernel
    scal_kernel_cpu<T>(nelems, alpha, &dst[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        T x = dst2[i];
        if(dst[i] != x*alpha)
        {
            throw std::runtime_error("Wrong value");
        }
    }
    // Now check StarPU codelet
    // StarPU interfaces
    StarpuVariableInterface dst2_interface(&dst2[0], nelems*sizeof(T));
    void *buffers[1] = {&dst2_interface};
    // Codelet arguments
    scal_starpu_args<T> args =
    {
        .nelems = nelems,
        .alpha = alpha
    };
    // Launch codelet
    scal_starpu_cpu<T>(buffers, &args);
    // Check it
    for(Index i = 0; i < nelems; ++i)
    {
        if(dst2[i] != dst[i])
        {
            throw std::runtime_error("Starpu codelet wrong result");
        }
    }
}

template<typename T>
void validate_many()
{
    Index nelems[3] = {0, 1, 80000};
    T alpha[4] = {-2, 0, 1, 11.1};
    for(Index i = 0; i < sizeof(nelems)/sizeof(nelems[0]); ++i)
    {
        for(Index j = 0; j < sizeof(alpha)/sizeof(alpha[0]); ++j)
        {
            validate<T>(nelems[i], alpha[j]);
        }
    }
}

int main(int argc, char **argv)
{
    validate_many<fp32_t>();
    validate_many<fp64_t>();
    return 0;
}

