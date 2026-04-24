/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/add_fiber.cc
 * Tile wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/add_fiber.hh"
#include "nntile/starpu/add_fiber.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Scalar beta, Index axis)
{
    using Y = typename T::repr_t;
    Tile<T> dst({3, 4, 5}), dst_ref({3, 4, 5});
    Tile<T> src1({dst.shape[axis]}), src2({3, 4, 5});
    auto s1 = src1.acquire(STARPU_W);
    auto s2 = src2.acquire(STARPU_W);
    auto d = dst.acquire(STARPU_W);
    auto dr = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        s1[i] = Y(i + 1);
    }
    for(Index i = 0; i < src2.nelems; ++i)
    {
        s2[i] = Y(0.5 * (i + 1));
        d[i] = Y(0);
        dr[i] = Y(0);
    }
    s1.release();
    s2.release();
    d.release();
    dr.release();

    Index m = dst.stride[axis];
    Index batch = src1.matrix_shape[1][1];
    Index n = dst.matrix_shape[axis+1][1] / batch;
    Index k = dst.shape[axis];
    starpu::add_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src1, beta,
            src2, dst);
    add_fiber<T>(alpha, src1, beta, src2, dst_ref, axis, 0);

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
    check<T>(1.0, 1.0, 2);
    check<T>(-0.5, 2.0, 0);
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
