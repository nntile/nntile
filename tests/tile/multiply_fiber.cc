/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/multiply_fiber.cc
 * Per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/multiply_fiber.hh"
#include "nntile/starpu/multiply_fiber.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check(Scalar alpha, Index axis)
{
    using Y = typename T::repr_t;
    Tile<T> dst({3, 4, 5}), dst_ref({3, 4, 5});
    Tile<T> src1({dst.shape[axis]}), src2({3, 4, 5});
    auto a = src1.acquire(STARPU_W);
    auto b = src2.acquire(STARPU_W);
    auto c = dst.acquire(STARPU_W);
    auto cr = dst_ref.acquire(STARPU_W);
    for(Index i = 0; i < src1.nelems; ++i)
    {
        a[i] = Y(0.5 + i);
    }
    for(Index i = 0; i < src2.nelems; ++i)
    {
        b[i] = Y(0.1 * (i + 1));
        c[i] = Y(0);
        cr[i] = Y(0);
    }
    a.release();
    b.release();
    c.release();
    cr.release();

    Index m = dst.stride[axis];
    Index n = dst.matrix_shape[axis+1][1];
    Index k = dst.shape[axis];
    starpu::multiply_fiber.submit<std::tuple<T>>(m, n, k, alpha, src1, src2,
            dst);
    multiply_fiber<T>(alpha, src1, src2, dst_ref, axis);

    c.acquire(STARPU_R);
    cr.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; ++i)
    {
        TEST_ASSERT(Y(c[i]) == Y(cr[i]));
    }
    c.release();
    cr.release();
}

template<typename T>
void validate()
{
    check<T>(1.0, 2);
    check<T>(-2.0, 0);
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
