/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/embedding_backward.cc
 * Backward embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/embedding_backward.hh"
#include "nntile/starpu/embedding_backward.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    constexpr Index m = 2, n = 2, k = 3, k_start = 0, k_size = 3;
    Tile<nntile::int64_t> index({m, n});
    Tile<T> embed_grad({m, k, n});
    Tile<T> vocab_grad_ref({k_size, 5});
    Tile<T> vocab_grad({k_size, 5});

    auto index_local = index.acquire(STARPU_W);
    index_local[0] = 0;
    index_local[1] = 2;
    index_local[2] = 4;
    index_local[3] = 1;
    index_local.release();

    auto embed_grad_local = embed_grad.acquire(STARPU_W);
    for(Index i = 0; i < embed_grad.nelems; ++i)
    {
        embed_grad_local[i] = Y(0.5 * (i+1));
    }
    embed_grad_local.release();

    auto vocab_grad_ref_local = vocab_grad_ref.acquire(STARPU_W);
    auto vocab_grad_local = vocab_grad.acquire(STARPU_W);
    for(Index i = 0; i < vocab_grad.nelems; ++i)
    {
        vocab_grad_ref_local[i] = Y(0.1 * (i+1));
        vocab_grad_local[i] = Y(vocab_grad_ref_local[i]);
    }
    vocab_grad_ref_local.release();
    vocab_grad_local.release();

    starpu::embedding_backward.submit<std::tuple<T>>(m, n, k, k_start, k_size,
            index, embed_grad, vocab_grad_ref, 0);
    embedding_backward<T>(m, n, k, k_start, k_size, index, embed_grad,
            vocab_grad, 0);

    vocab_grad_ref_local.acquire(STARPU_R);
    vocab_grad_local.acquire(STARPU_R);
    for(Index i = 0; i < vocab_grad.nelems; ++i)
    {
        TEST_ASSERT(Y(vocab_grad_ref_local[i]) == Y(vocab_grad_local[i]));
    }
    vocab_grad_ref_local.release();
    vocab_grad_local.release();
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
