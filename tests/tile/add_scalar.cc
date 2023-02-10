/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/add_scalar.cc
 * Add scalar to elements from Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/tile/add_scalar.hh"
#include "nntile/starpu/add_scalar.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate(T val, const std::vector<Index>& n_elements)
{
    Tile<T> src1(n_elements);
    Tile<T> src1_copy(n_elements);
    auto src1_local = src1.acquire(STARPU_W);
    auto src1_copy_local = src1_copy.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        src1_local[i] = T(i+1);
        src1_copy_local[i] = T(i+ 1);
    }
    src1_local.release();
    src1_copy_local.release();
    
    
    starpu::add_scalar::submit<T>(val, src1.nelems, src1);
    add_scalar<T>(val, src1_copy);

    src1_local.acquire(STARPU_R);
    src1_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        TEST_ASSERT(src1_local[i] == src1_copy_local[i]);
    }
    src1_local.release();
    src1_copy_local.release();
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::add_scalar::init();
    starpu::add_scalar::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>(10, {2});
    validate<fp64_t>(10, {2});
    return 0;
}
