/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/pow.cc
 * Power operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/pow.hh"
#include "nntile/starpu/pow.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    Tile<T> a({2, 3, 4}), b({2, 3, 4});
    auto al = a.acquire(STARPU_W);
    auto bl = b.acquire(STARPU_W);
    for(Index i = 0; i < a.nelems; ++i)
    {
        al[i] = Y(0.5 + 0.1 * i);
        bl[i] = al[i];
    }
    al.release();
    bl.release();
    Scalar alpha = 2.0, exp = 1.5;
    starpu::pow.submit<std::tuple<T>>(a.nelems, alpha, exp, a);
    pow<T>(alpha, exp, b);
    al.acquire(STARPU_R);
    bl.acquire(STARPU_R);
    for(Index i = 0; i < a.nelems; ++i)
    {
        TEST_ASSERT(Y(al[i]) == Y(bl[i]));
    }
    al.release();
    bl.release();
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
