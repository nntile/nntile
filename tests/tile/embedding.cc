/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/embedding.cc
 * Embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tile/embedding.hh"
#include "nntile/starpu/embedding.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void validate()
{
    using Y = typename T::repr_t;
    constexpr Index m = 2, n = 2, k = 3, k_start = 0, k_size = 3;
    Tile<nntile::int64_t> index({m, n});
    Tile<T> vocab({k_size, 5});
    Tile<T> embed_ref({m, k, n});
    Tile<T> embed({m, k, n});

    auto index_local = index.acquire(STARPU_W);
    index_local[0] = 0;
    index_local[1] = 2;
    index_local[2] = 4;
    index_local[3] = 1;
    index_local.release();

    auto vocab_local = vocab.acquire(STARPU_W);
    for(Index i = 0; i < vocab.nelems; ++i)
    {
        vocab_local[i] = Y(i+1);
    }
    vocab_local.release();

    auto embed_ref_local = embed_ref.acquire(STARPU_W);
    auto embed_local = embed.acquire(STARPU_W);
    for(Index i = 0; i < embed.nelems; ++i)
    {
        embed_ref_local[i] = Y(0);
        embed_local[i] = Y(0);
    }
    embed_ref_local.release();
    embed_local.release();

    starpu::embedding.submit<std::tuple<T>>(m, n, k, k_start, k_size, index,
            vocab, embed_ref);
    embedding<T>(m, n, k, k_start, k_size, index, vocab, embed);

    embed_ref_local.acquire(STARPU_R);
    embed_local.acquire(STARPU_R);
    for(Index i = 0; i < embed.nelems; ++i)
    {
        TEST_ASSERT(Y(embed_ref_local[i]) == Y(embed_local[i]));
    }
    embed_ref_local.release();
    embed_local.release();
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
