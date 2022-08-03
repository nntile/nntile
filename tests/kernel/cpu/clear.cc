/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/clear.cc
 * Clear a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-02
 * */

#include "nntile/kernel/cpu/clear.hh"
#include "nntile/starpu.hh"
#include "nntile/base_types.hh"
#include <vector>
#include <stdexcept>

using namespace nntile;

// Templated validation
template<typename T>
void validate(Index m)
{
    // Init test input
    std::vector<T> dst(m);
    for(Index i = 0; i < m; ++i)
    {
        dst[i] = T(i+1) / T{10};
    }
    std::vector<T> dst2(dst);
    // Check low-level kernel
    clear_kernel_cpu(m*sizeof(T), &dst[0]);
    for(Index i = 0; i < m; ++i)
    {
        if(dst[i] != T{0})
        {
            throw std::runtime_error("Wrong value");
        }
    }
    // Now check StarPU codelet
    // StarPU interfaces
    StarpuVariableInterface dst2_interface(&dst2[0], m*sizeof(T));
    void *buffers[1] = {&dst2_interface};
    // No codelet arguments
    // Launch codelet
    clear_starpu_cpu(buffers, nullptr);
    // Check it
    for(Index i = 0; i < m; ++i)
    {
        if(dst2[i] != T{0})
        {
            throw std::runtime_error("Starpu codelet wrong result");
        }
    }
}

int main(int argc, char **argv)
{
    validate<fp32_t>(0);
    validate<fp32_t>(1);
    validate<fp32_t>(80000);
    validate<fp64_t>(0);
    validate<fp64_t>(1);
    validate<fp64_t>(80000);
    return 0;
}

