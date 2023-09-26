/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/flash_maxsumexp.cc
 * Fast max and sum of exponents of Tensor<T> along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-24
 * */

#include "nntile/tensor/flash_maxsumexp.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/starpu/mask_scalar.hh"
#include "nntile/starpu/maxsumexp.hh"
#include <cmath>
#include <limits>

namespace nntile
{
namespace tensor
{

//! Compute max and sum of exponents of slices along given axis
template<typename T>
void flash_maxsumexp_async(const Tensor<T> &Q, const Tensor<T> &K,
        const Tensor<bool_t> &mask, const Tensor<T> &maxsumexp,
        const Tensor<T> &tmp, int redux)
{
//    // Check dimensions
//    if(src.ndim != dst.ndim)
//    {
//        throw std::runtime_error("src.ndim != dst.ndim");
//    }
//    // Treat special case of src.ndim=0
//    if(src.ndim == 0)
//    {
//        throw std::runtime_error("Scalar input makes no sense");
//    }
//    // Check axis
//    if(axis < 0)
//    {
//        throw std::runtime_error("axis < 0");
//    }
//    if(axis >= src.ndim)
//    {
//        throw std::runtime_error("axis >= src.ndim");
//    }
//    // Check shapes of src and dst
//    if(dst.shape[0] != 2)
//    {
//        throw std::runtime_error("dst.shape[0] != 2");
//    }
//    if(dst.basetile_shape[0] != 2)
//    {
//        throw std::runtime_error("dst.basetile_shape[0] != 2");
//    }
//    for(Index i = 0; i < axis; ++i)
//    {
//        if(src.shape[i] != dst.shape[i+1])
//        {
//            throw std::runtime_error("src.shape[i] != dst.shape[i+1]");
//        }
//        if(src.basetile_shape[i] != dst.basetile_shape[i+1])
//        {
//            throw std::runtime_error("src.basetile_shape[i] != "
//                    "dst.basetile_shape[i+1]");
//        }
//    }
//    for(Index i = axis+1; i < src.ndim; ++i)
//    {
//        if(src.shape[i] != dst.shape[i])
//        {
//            throw std::runtime_error("src.shape[i] != dst.shape[i]");
//        }
//        if(src.basetile_shape[i] != dst.basetile_shape[i])
//        {
//            throw std::runtime_error("src.basetile_shape[i] != "
//                    "dst.basetile_shape[i]");
//        }
//    }
    // Do actual calculations
    int ret;
    Index head_size = Q.shape[0];
    Index n_seq_tile = Q.basetile_shape[1];
    Index n_batch_tile = Q.basetile_shape[2];
    Index n_head_tile = Q.basetile_shape[3];
    const TransOp opT(TransOp::Trans), opN(TransOp::NoTrans);
    for(Index i = 0; i < maxsumexp.grid.nelems; ++i)
    {
        // Destination tile on dest node must be already prepared (cleared)
        auto maxsumexp_tile_handle = maxsumexp.get_tile_handle(i);
        // Obtain indices of applicable source tiles
        auto maxsumexp_tile_index = maxsumexp.grid.linear_to_index(i);
        std::vector<Index> tmp_tile_index(maxsumexp_tile_index),
            q_tile_index(maxsumexp_tile_index),
            k_tile_index(maxsumexp_tile_index),
            mask_tile_index(2);
        auto q_tile_handle = Q.get_tile_handle(q_tile_index);
        mask_tile_index[1] = maxsumexp_tile_index[1];
        // Launch kernel for each appropriate tile of K to accumulate maxsumexp
        // result
        for(Index j = 0; j < K.grid.shape[1]; ++j)
        {
            tmp_tile_index[0] = j;
            k_tile_index[1] = j;
            mask_tile_index[0] = j;
            auto tmp_tile_handle = tmp.get_tile_handle(tmp_tile_index);
            auto k_tile_handle = K.get_tile_handle(k_tile_index);
            auto mask_tile_handle = mask.get_tile_handle(mask_tile_index);
            // Insert tasks
            starpu::gemm::submit<T, T>(opT, opN,
                    n_seq_tile, n_seq_tile, head_size,
                    n_batch_tile*n_head_tile, 1.0/std::sqrt(head_size),
                    k_tile_handle, q_tile_handle, 0.0, tmp_tile_handle,
                    redux=0);
            starpu::mask_scalar::submit<T>(n_seq_tile*n_seq_tile,
                    n_batch_tile*n_head_tile, mask_tile_handle,
                    -std::numeric_limits<T>::infinity(), tmp_tile_handle);
            starpu::maxsumexp::submit<T>(1,
                    n_seq_tile*n_batch_tile*n_head_tile, n_seq_tile,
                    tmp_tile_handle, maxsumexp_tile_handle, redux=0);
        }
    }
}

template<typename T>
void flash_maxsumexp(const Tensor<T> &Q, const Tensor<T> &K,
        const Tensor<bool_t> &mask, const Tensor<T> &maxsumexp,
        const Tensor<T> &tmp, int redux)
{
    flash_maxsumexp_async<T>(Q, K, mask, maxsumexp, tmp, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void flash_maxsumexp_async(const Tensor<fp32_t> &Q, const Tensor<fp32_t> &K,
        const Tensor<bool_t> &mask, const Tensor<fp32_t> &maxsumexp,
        const Tensor<fp32_t> &tmp, int redux);

template
void flash_maxsumexp_async(const Tensor<fp64_t> &Q, const Tensor<fp64_t> &K,
        const Tensor<bool_t> &mask, const Tensor<fp64_t> &maxsumexp,
        const Tensor<fp64_t> &tmp, int redux);

// Explicit instantiation
template
void flash_maxsumexp(const Tensor<fp32_t> &Q, const Tensor<fp32_t> &K,
        const Tensor<bool_t> &mask, const Tensor<fp32_t> &maxsumexp,
        const Tensor<fp32_t> &tmp, int redux);

template
void flash_maxsumexp(const Tensor<fp64_t> &Q, const Tensor<fp64_t> &K,
        const Tensor<bool_t> &mask, const Tensor<fp64_t> &maxsumexp,
        const Tensor<fp64_t> &tmp, int redux);

} // namespace tensor
} // namespace nntile

