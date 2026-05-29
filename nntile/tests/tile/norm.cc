/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/norm.cc
 * Euclidean norm of all elements in a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/norm.hh"
#include "nntile/starpu/norm.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    const Y eps = T::epsilon;
    // Create tiles
    Tile<T> src({10});
    Tile<T> dst({});

    // Fill source tile
    auto src_local = src.acquire(STARPU_W);
    for(Index i = 0; i < src.nelems; ++i)
    {
        src_local[i] = Y(1.0);
    }
    src_local.release();

    // Fill destination tile
    auto dst_local = dst.acquire(STARPU_W);
    dst_local[0] = Y(0.0);
    dst_local.release();

    // Run norm operation
    norm_async<T>(1.0, src, 0.0, dst);
    starpu_task_wait_for_all();

    // Check result
    auto dst_check = dst.acquire(STARPU_R);
    Y expected = std::sqrt(Y(10.0));
    if(std::abs(static_cast<Y>(dst_check[0]) - expected) > eps)
    {
        throw std::runtime_error("Tile norm validation failed");
    }
    dst_check.release();
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    int ncpu=1, ncuda=1, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate<fp64_t>();
    validate<fp32_t>();
    validate<bf16_t>();
    validate<fp16_t>();
    return 0;
}
