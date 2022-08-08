/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/gelutanh.cc
 * Approximate GeLU operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-04
 * */

#include "nntile/kernel/cpu/gelutanh.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

using namespace nntile;
using namespace nntile::kernel::cpu;

// Templated validation
template<typename T>
void validate(Index nelems)
{
    constexpr T pi = 3.141592653589793238462643383279502884L;
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = T(2*i+1-nelems) / T{10};
    }
    std::vector<T> dst2(dst);
    // Check low-level kernel
    gelutanh<T>(nelems, &dst[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        T x = dst2[i];
        T y = std::sqrt(T{2}/pi) * (x+T{0.044715}*x*x*x);
        T z = T{1}+std::tanh(y);
        T val_ref = T{0.5} * x * z;
        // Obtain range of correct values
        T val_ref_min, val_ref_max;
        if(val_ref < 0)
        {
            val_ref_min = val_ref * (T{1}+eps) - eps;
            val_ref_max = val_ref * (T{1}-eps) + eps;
        }
        else
        {
            val_ref_min = val_ref * (T{1}-eps) - eps;
            val_ref_max = val_ref * (T{1}+eps) + eps;
        }
        if(dst[i] < val_ref_min or dst[i] > val_ref_max)
        {
            throw std::runtime_error("Wrong value");
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

