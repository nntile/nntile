/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/norm_fiber.cc
 * Euclidean norms over slices into a fiber of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/norm_fiber.hh"
#include "nntile/starpu/norm_fiber.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    const Y eps = T::epsilon();
    // constants
    Index batch = 5;
    Index m = 3;
    Index n = 20;
    Index k = 1;
    Scalar alpha = 1.0, beta = 0.0;
    int redux = 0;
    Index batch_ndim = 0;
    int axis = 0;
    // tiles
    Tile<T> src1({batch, m, n, k});
    Tile<T> src2({batch});
    Tile<T> dst({batch});

    // fill tiles
    auto src1_local = src1.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        src1_local[i] = Y(-1.0);
    }
    src1_local.release();

    auto src2_local = src2.acquire(STARPU_W);
    for(Index i = 0; i < src2.nelems; ++i)
    {
        src2_local[i] = Y(1.0);
    }
    src2_local.release();
    
    
    auto dst_local_w = dst.acquire(STARPU_W);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        dst_local_w[i] = Y(-1.0);
    }
    dst_local_w.release();

    std::cout << "Run tile::norm_fiber<" << T::type_repr << "> restricted to CPU\n";
    norm_fiber<T>(alpha, src1, beta, src2, dst, axis, batch_ndim, redux);
    auto dst_local = dst.acquire(STARPU_R);
    auto val = Y(dst_local[0]);
    auto ref = Y(std::sqrt((m*n*k)));
    TEST_ASSERT(std::abs(val/ref-Y{1}) <= 10*eps)
    std::cout << "OK: tile::norm_fiber<" << T::type_repr << "> restricted to CPU\n";
    dst_local.release();
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::norm_fiber::init();
    starpu::norm_fiber::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
