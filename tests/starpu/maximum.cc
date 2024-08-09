/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/maximum.cc
 * Elementwise maximum operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/maximum.hh"
#include "nntile/starpu/maximum.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> src1({}), dst1({}), dst1_copy({}), src2({2, 3, 4}),
        dst2({2, 3, 4}), dst2_copy({2, 3, 4});
    auto src1_local = src1.acquire(STARPU_W);
    auto dst1_local = dst1.acquire(STARPU_W);
    auto dst1_copy_local = dst1_copy.acquire(STARPU_W);
    src1_local[0] = Y{2};
    dst1_local[0] = Y{-1};
    dst1_copy_local[0] = Y{-1};
    src1_local.release();
    dst1_local.release();
    dst1_copy_local.release();
    auto src2_local = src2.acquire(STARPU_W);
    auto dst2_local = dst2.acquire(STARPU_W);
    auto dst2_copy_local = dst2_copy.acquire(STARPU_W);
    for(Index i = 0; i < src2.nelems; ++i)
    {
        src2_local[i] = Y(i+1);
        dst2_local[i] = Y(i-10);
        dst2_copy_local[i] = Y(i-10);
    }
    src2_local.release();
    dst2_local.release();
    dst2_copy_local.release();
    starpu::maximum::submit<T>(1, src1, dst1);
    maximum<T>(src1, dst1_copy);
    dst1_local.acquire(STARPU_R);
    dst1_copy_local.acquire(STARPU_R);
    TEST_ASSERT(Y(dst1_local[0]) == Y(dst1_copy_local[0]));
    dst1_local.release();
    dst1_copy_local.release();
    starpu::maximum::submit<T>(src2.nelems, src2, dst2);
    maximum<T>(src2, dst2_copy);
    dst2_local.acquire(STARPU_R);
    dst2_copy_local.acquire(STARPU_R);
    for(Index i = 0; i < src2.nelems; ++i)
    {
        TEST_ASSERT(Y(dst2_local[i]) == Y(dst2_copy_local[i]));
    }
    dst2_local.release();
    dst2_copy_local.release();
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::maximum::init();
    starpu::maximum::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
