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

#include "nntile/context.hh"
#include "nntile/tile/norm_fiber.hh"
#include "nntile/starpu/norm_fiber.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    Scalar alpha = 1.0, beta = 0.0;
    int redux = 0;
    Index batch_ndim = 0;
    int axis = 0;
    Tile<T> src1({5, 3, 20, 1});
    Tile<T> src2({5});
    Tile<T> dst({5});
    Tile<T> dst_ref({5});
    auto s1 = src1.acquire(STARPU_W);
    auto s2 = src2.acquire(STARPU_W);
    auto d = dst.acquire(STARPU_W);
    auto dr = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        s1[i] = Y(-1.0);
    }
    for(Index i = 0; i < src2.nelems; ++i)
    {
        s2[i] = Y(0.0);
    }
    for(Index i = 0; i < dst.nelems; ++i)
    {
        d[i] = Y(0.0);
        dr[i] = Y(0.0);
    }
    s1.release();
    s2.release();
    d.release();
    dr.release();

    Index batch = dst.matrix_shape[1][1];
    Index m = src1.stride[axis];
    Index n = src1.matrix_shape[axis+1][1] / batch;
    Index k = src1.shape[axis];
    starpu::norm_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src1, beta,
            src2, dst, redux);
    norm_fiber<T>(alpha, src1, beta, src2, dst_ref, axis, batch_ndim, redux);

    d.acquire(STARPU_R);
    dr.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(d[i]) == Y(dr[i]));
    }
    d.release();
    dr.release();
}

template<typename T>
void validate()
{
    check<T>();
}

int main(int argc, char **argv)
{
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
