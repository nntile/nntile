/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/embedding_backward.cc
 * Backward embeddings from vocabulary within Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/embedding_backward.hh"
#include "nntile/starpu/embedding_backward.hh"

namespace nntile::tensor
{

template<typename T>
void embedding_backward_async(const Tensor<int64_t> &index,
        const Tensor<T> &embed, const Tensor<T> &vocab, Index axis, int redux)
{
    // Check dimensions
    if(index.ndim+1 != embed.ndim)
    {
        throw std::runtime_error("index.ndim+1 != embed.ndim");
    }
    if(vocab.ndim != 2)
    {
        throw std::runtime_error("vocab.ndim != 2");
    }
    // Check shapes
    for(Index i = 0; i < axis-1; ++i)
    {
        if(index.shape[i] != embed.shape[i])
        {
            throw std::runtime_error("index.shape[i] != embed.shape[i]");
        }
        if(index.basetile_shape[i] != embed.basetile_shape[i])
        {
            throw std::runtime_error("index.basetile_shape[i] != "
                    "embed.basetile_shape[i]");
        }
    }
    for(Index i = axis; i < index.ndim; ++i)
    {
        if(index.shape[i] != embed.shape[i+1])
        {
            throw std::runtime_error("index.shape[i] != embed.shape[i+1]");
        }
        if(index.basetile_shape[i] != embed.basetile_shape[i+1])
        {
            throw std::runtime_error("index.basetile_shape[i] != "
                    "embed.basetile_shape[i+1]");
        }
    }
    if(embed.shape[axis] != vocab.shape[0])
    {
        throw std::runtime_error("embed.shape[axis] != vocab.shape[0]");
    }
    if(embed.basetile_shape[axis] % vocab.basetile_shape[0] != 0)
    {
        throw std::runtime_error("embed.basetile_shape[axis] % "
                "vocab.basetile_shape[0] != 0");
    }
    // Actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    // Cycle over embedding tiles
    for(Index i = 0; i < embed.grid.nelems; ++i)
    {
        auto embed_tile_handle = embed.get_tile_handle(i);
        auto embed_tile_traits = embed.get_tile_traits(i);
        int embed_tile_rank = embed_tile_handle.mpi_get_rank();
        auto embed_tile_index = embed.grid.linear_to_index(i);
        // Get corresponding index tile
        std::vector<Index> index_tile_index(index.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            index_tile_index[j] = embed_tile_index[j];
        }
        for(Index j = axis; j < index.ndim; ++j)
        {
            index_tile_index[j] = embed_tile_index[j+1];
        }
        auto index_tile_handle = index.get_tile_handle(index_tile_index);
        int index_tile_rank = index_tile_handle.mpi_get_rank();
        // Number of vocab tiles per single embed tile
        Index vocab_per_embed = (embed_tile_traits.shape[axis]-1)
            / vocab.basetile_shape[0] + 1;
        // Find corresponding vocab tiles and copy embeddings
        Index vocab_start = embed_tile_index[axis] * embed.basetile_shape[axis]
            / vocab.basetile_shape[0];
        Index vocab_end = vocab_start + vocab_per_embed;
        for(Index j = vocab_start; j < vocab_end; ++j)
        {
            auto vocab_tile_handle = vocab.get_tile_handle(j);
            auto vocab_tile_traits = vocab.get_tile_traits(j);
            Index m, n, k, k_start, k_size;
            m = embed_tile_traits.stride[axis];
            n = embed_tile_traits.matrix_shape[axis+1][1];
            k = embed_tile_traits.shape[axis];
            k_start = (j-vocab_start) * vocab.basetile_shape[0];
            k_size = vocab_tile_traits.shape[0];
            starpu::embedding_backward::submit<T>(m, n, k, k_start, k_size,
                    index_tile_handle, embed_tile_handle, vocab_tile_handle,
                    redux);
        }
    }
    // Flush cache for the output tile on every node
    for(Index i = 0; i < vocab.grid.nelems; ++i)
    {
        vocab.get_tile_handle(i).mpi_flush();
    }
}

template<typename T>
void embedding_backward(const Tensor<int64_t> &index, const Tensor<T> &embed,
        const Tensor<T> &vocab, Index axis, int redux)
{
    embedding_backward_async<T>(index, embed, vocab, axis, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void embedding_backward_async<fp32_t>(const Tensor<int64_t> &index,
        const Tensor<fp32_t> &embed, const Tensor<fp32_t> &vocab, Index axis,
        int redux);

template
void embedding_backward_async<fp32_fast_tf32_t>(const Tensor<int64_t> &index,
        const Tensor<fp32_fast_tf32_t> &embed, const Tensor<fp32_fast_tf32_t> &vocab, Index axis,
        int redux);

template
void embedding_backward_async<fp64_t>(const Tensor<int64_t> &index,
        const Tensor<fp64_t> &embed, const Tensor<fp64_t> &vocab, Index axis,
        int redux);

template
void embedding_backward_async<bf16_t>(const Tensor<int64_t> &index,
        const Tensor<bf16_t> &embed, const Tensor<bf16_t> &vocab, Index axis,
        int redux);

// Explicit instantiation
template
void embedding_backward<fp32_t>(const Tensor<int64_t> &index,
        const Tensor<fp32_t> &embed, const Tensor<fp32_t> &vocab, Index axis,
        int redux);

template
void embedding_backward<fp32_fast_tf32_t>(const Tensor<int64_t> &index,
        const Tensor<fp32_fast_tf32_t> &embed, const Tensor<fp32_fast_tf32_t> &vocab, Index axis,
        int redux);

template
void embedding_backward<fp64_t>(const Tensor<int64_t> &index,
        const Tensor<fp64_t> &embed, const Tensor<fp64_t> &vocab, Index axis,
        int redux);

template
void embedding_backward<bf16_t>(const Tensor<int64_t> &index,
        const Tensor<bf16_t> &embed, const Tensor<bf16_t> &vocab, Index axis,
        int redux);

} // namespace nntile::tensor
