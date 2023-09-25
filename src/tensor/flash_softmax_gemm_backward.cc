/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/flash_softmax_gemm_backward.cc
 * Fast backward of softmax and gemm operations
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-25
 * */

#include "nntile/tensor/flash_softmax_gemm_backward.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/starpu/mask_scalar.hh"
#include "nntile/starpu/softmax_inplace.hh"
#include "nntile/starpu/sumprod_slice.hh"
#include "nntile/starpu/add_slice.hh"
#include "nntile/starpu/prod.hh"

namespace nntile
{
namespace tensor
{

template<typename T>
void flash_softmax_gemm_backward_async(const Tensor<T> &Q, const Tensor<T> &dQ,
        const Tensor<T> &K, const Tensor<T> &dK, const Tensor<T> &V,
        const Tensor<T> &dV, const Tensor<bool_t> &mask,
        const Tensor<T> &maxsumexp, const Tensor<T> &dst_grad,
        const Tensor<T> &tmp, const Tensor<T> &tmp_grad,
        const Tensor<T> &tmp_sumprod_slice, int redux)
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
    // Cycle for all tiles of dV tensor
    for(Index i = 0; i < dV.grid.nelems; ++i)
    {
        auto dQ_tile_handle = dQ.get_tile_handle(i);
        auto dK_tile_handle = dK.get_tile_handle(i);
        auto dV_tile_handle = dV.get_tile_handle(i);
        auto dV_tile_index = dV.grid.linear_to_index(i);
        // Indices of all required tensors
        std::vector<Index> tmp_tile_index(dV_tile_index),
            q_tile_index(dV_tile_index), dq_tile_index(dV_tile_index),
            k_tile_index(dV_tile_index), dk_tile_index(dV_tile_index),
            v_tile_index(dV_tile_index), dst_grad_tile_index(dV_tile_index),
            mask_tile_index(2), maxsumexp_tile_index(dV_tile_index),
            tmp_grad_tile_index(dV_tile_index),
            tmp_sumprod_slice_tile_index(3);
        auto k_tile_handle = K.get_tile_handle(k_tile_index);
        auto v_tile_handle = V.get_tile_handle(v_tile_index);
        tmp_tile_index[0] = dV_tile_index[1];
        tmp_grad_tile_index[0] = dV_tile_index[1];
        tmp_sumprod_slice_tile_index[1] = dV_tile_index[2];
        tmp_sumprod_slice_tile_index[2] = dV_tile_index[3];
        mask_tile_index[0] = dV_tile_index[1];
        // Clear destination buffers at first
        starpu::clear::submit(dQ_tile_handle);
        starpu::clear::submit(dK_tile_handle);
        starpu::clear::submit(dV_tile_handle);
        // Launch kernel for each appropriate tile of K and V to accumulate
        // result into destination tensor
        for(Index j = 0; j < Q.grid.shape[1]; ++j)
        {
            tmp_tile_index[1] = j;
            tmp_grad_tile_index[1] = j;
            q_tile_index[1] = j;
            dst_grad_tile_index[1] = j;
            mask_tile_index[1] = j;
            maxsumexp_tile_index[1] = j;
            tmp_sumprod_slice_tile_index[0] = j;
            auto tmp_tile_handle = tmp.get_tile_handle(tmp_tile_index);
            auto tmp_grad_tile_handle = tmp_grad.get_tile_handle(
                    tmp_grad_tile_index);
            auto tmp_sumprod_slice_tile_handle = tmp_sumprod_slice
                .get_tile_handle(tmp_sumprod_slice_tile_index);
            auto q_tile_handle = Q.get_tile_handle(q_tile_index);
            auto dst_grad_tile_handle = dst_grad.get_tile_handle(
                    dst_grad_tile_index);
            auto mask_tile_handle = mask.get_tile_handle(mask_tile_index);
            auto maxsumexp_tile_handle = maxsumexp.get_tile_handle(
                    maxsumexp_tile_index);
            // Insert tasks to accumulate dV
            starpu::gemm::submit<T, T>(opT, opN,
                    n_seq_tile, n_seq_tile, head_size,
                    n_batch_tile*n_head_tile, 1.0/std::sqrt(head_size),
                    k_tile_handle, q_tile_handle, 0.0, tmp_tile_handle,
                    redux=0);
            starpu::mask_scalar::submit<T>(n_seq_tile*n_seq_tile,
                    n_batch_tile*n_head_tile, mask_tile_handle,
                    -std::numeric_limits<T>::infinity(), tmp_tile_handle);
            starpu::softmax_inplace::submit<T>(1,
                    n_seq_tile*n_batch_tile*n_head_tile, n_seq_tile,
                    maxsumexp_tile_handle, tmp_tile_handle);
            starpu::gemm::submit<T, T>(opN, opT,
                    head_size, n_seq_tile, n_seq_tile,
                    n_batch_tile*n_head_tile, 1.0,
                    dst_grad_tile_handle, tmp_tile_handle, 1.0, dV_tile_handle,
                    redux=0);
            // Insert tasks to compute sumprod_slice(V'@dst_grad, softmax(...))
            starpu::gemm::submit<T, T>(opT, opN,
                    n_seq_tile, n_seq_tile, head_size,
                    n_batch_tile*n_head_tile, 1.0,
                    v_tile_handle, dst_grad_tile_handle, 0.0,
                    tmp_grad_tile_handle, redux=0);
            starpu::sumprod_slice::submit<T>(1,
                    n_seq_tile*n_batch_tile*n_head_tile, n_seq_tile,
                    1.0, tmp_grad_tile_handle, tmp_tile_handle,
                    1.0, tmp_sumprod_slice_tile_handle, redux=0);
        }
    }
    // Cycle for all tiles of dK/dV tensor
    for(Index i = 0; i < dV.grid.nelems; ++i)
    {
        auto dK_tile_handle = dK.get_tile_handle(i);
        auto dV_tile_index = dV.grid.linear_to_index(i);
        // Indices of all required tensors
        std::vector<Index> tmp_tile_index(dV_tile_index),
            q_tile_index(dV_tile_index), dq_tile_index(dV_tile_index),
            k_tile_index(dV_tile_index), dk_tile_index(dV_tile_index),
            v_tile_index(dV_tile_index), dst_grad_tile_index(dV_tile_index),
            mask_tile_index(2), maxsumexp_tile_index(dV_tile_index),
            tmp_grad_tile_index(dV_tile_index),
            tmp_sumprod_slice_tile_index(3);
        auto k_tile_handle = K.get_tile_handle(k_tile_index);
        auto v_tile_handle = V.get_tile_handle(v_tile_index);
        tmp_tile_index[0] = dV_tile_index[1];
        tmp_grad_tile_index[0] = dV_tile_index[1];
        tmp_sumprod_slice_tile_index[1] = dV_tile_index[2];
        tmp_sumprod_slice_tile_index[2] = dV_tile_index[3];
        mask_tile_index[0] = dV_tile_index[1];
        for(Index j = 0; j < dQ.grid.shape[1]; ++j)
        {
            tmp_tile_index[1] = j;
            tmp_grad_tile_index[1] = j;
            q_tile_index[1] = j;
            dst_grad_tile_index[1] = j;
            mask_tile_index[1] = j;
            maxsumexp_tile_index[1] = j;
            tmp_sumprod_slice_tile_index[0] = j;
            dq_tile_index[1] = j;
            auto tmp_tile_handle = tmp.get_tile_handle(tmp_tile_index);
            auto tmp_grad_tile_handle = tmp_grad.get_tile_handle(
                    tmp_grad_tile_index);
            auto tmp_sumprod_slice_tile_handle = tmp_sumprod_slice
                .get_tile_handle(tmp_sumprod_slice_tile_index);
            auto q_tile_handle = Q.get_tile_handle(q_tile_index);
            auto dst_grad_tile_handle = dst_grad.get_tile_handle(
                    dst_grad_tile_index);
            auto mask_tile_handle = mask.get_tile_handle(mask_tile_index);
            auto maxsumexp_tile_handle = maxsumexp.get_tile_handle(
                    maxsumexp_tile_index);
            auto dQ_tile_handle = dQ.get_tile_handle(dq_tile_index);
            // Insert tasks
            starpu::gemm::submit<T, T>(opT, opN,
                    n_seq_tile, n_seq_tile, head_size,
                    n_batch_tile*n_head_tile, 1.0,
                    v_tile_handle, dst_grad_tile_handle, 0.0,
                    tmp_grad_tile_handle, redux=0);
            starpu::add_slice::submit<T>(1,
                    n_seq_tile*n_batch_tile*n_head_tile, n_seq_tile,
                    -1.0, tmp_sumprod_slice_tile_handle,
                    1.0, tmp_grad_tile_handle);
            starpu::gemm::submit<T, T>(opT, opN,
                    n_seq_tile, n_seq_tile, head_size,
                    n_batch_tile*n_head_tile, 1.0/std::sqrt(head_size),
                    k_tile_handle, q_tile_handle, 0.0, tmp_tile_handle,
                    redux=0);
            starpu::mask_scalar::submit<T>(n_seq_tile*n_seq_tile,
                    n_batch_tile*n_head_tile, mask_tile_handle,
                    -std::numeric_limits<T>::infinity(), tmp_tile_handle);
            starpu::softmax_inplace::submit<T>(1,
                    n_seq_tile*n_batch_tile*n_head_tile, n_seq_tile,
                    maxsumexp_tile_handle, tmp_tile_handle);
            starpu::prod::submit<T>(
                    n_seq_tile*n_seq_tile*n_batch_tile*n_head_tile,
                    tmp_tile_handle, tmp_grad_tile_handle);
            starpu::mask_scalar::submit<T>(n_seq_tile*n_seq_tile,
                    n_batch_tile*n_head_tile, mask_tile_handle,
                    0.0, tmp_grad_tile_handle);
            starpu::gemm::submit<T, T>(opN, opN,
                    head_size, n_seq_tile, n_seq_tile,
                    n_batch_tile*n_head_tile, 1.0/std::sqrt(head_size),
                    k_tile_handle, tmp_grad_tile_handle, 1.0, dQ_tile_handle,
                    redux=0);
            starpu::gemm::submit<T, T>(opN, opT,
                    head_size, n_seq_tile, n_seq_tile,
                    n_batch_tile*n_head_tile, 1.0/std::sqrt(head_size),
                    q_tile_handle, tmp_grad_tile_handle, 1.0, dK_tile_handle,
                    redux=0);
        }
    }
}

template<typename T>
void flash_softmax_gemm_backward(const Tensor<T> &Q, const Tensor<T> &dQ,
        const Tensor<T> &K, const Tensor<T> &dK, const Tensor<T> &V,
        const Tensor<T> &dV, const Tensor<bool_t> &mask,
        const Tensor<T> &maxsumexp, const Tensor<T> &dst_grad,
        const Tensor<T> &tmp, const Tensor<T> &tmp_grad,
        const Tensor<T> &tmp_sumprod_slice, int redux)
{
    flash_softmax_gemm_backward_async<T>(Q, dQ, K, dK, V, dV, mask, maxsumexp,
            dst_grad, tmp, tmp_grad, tmp_sumprod_slice, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void flash_softmax_gemm_backward_async(const Tensor<fp32_t> &Q, const Tensor<fp32_t> &dQ,
        const Tensor<fp32_t> &K, const Tensor<fp32_t> &dK, const Tensor<fp32_t> &V,
        const Tensor<fp32_t> &dV, const Tensor<bool_t> &mask,
        const Tensor<fp32_t> &maxsumexp, const Tensor<fp32_t> &dst_grad,
        const Tensor<fp32_t> &tmp, const Tensor<fp32_t> &tmp_grad,
        const Tensor<fp32_t> &tmp_sumprod_slice, int redux);

template
void flash_softmax_gemm_backward_async(const Tensor<fp64_t> &Q, const Tensor<fp64_t> &dQ,
        const Tensor<fp64_t> &K, const Tensor<fp64_t> &dK, const Tensor<fp64_t> &V,
        const Tensor<fp64_t> &dV, const Tensor<bool_t> &mask,
        const Tensor<fp64_t> &maxsumexp, const Tensor<fp64_t> &dst_grad,
        const Tensor<fp64_t> &tmp, const Tensor<fp64_t> &tmp_grad,
        const Tensor<fp64_t> &tmp_sumprod_slice, int redux);

// Explicit instantiation
template
void flash_softmax_gemm_backward(const Tensor<fp32_t> &Q, const Tensor<fp32_t> &dQ,
        const Tensor<fp32_t> &K, const Tensor<fp32_t> &dK, const Tensor<fp32_t> &V,
        const Tensor<fp32_t> &dV, const Tensor<bool_t> &mask,
        const Tensor<fp32_t> &maxsumexp, const Tensor<fp32_t> &dst_grad,
        const Tensor<fp32_t> &tmp, const Tensor<fp32_t> &tmp_grad,
        const Tensor<fp32_t> &tmp_sumprod_slice, int redux);

template
void flash_softmax_gemm_backward(const Tensor<fp64_t> &Q, const Tensor<fp64_t> &dQ,
        const Tensor<fp64_t> &K, const Tensor<fp64_t> &dK, const Tensor<fp64_t> &V,
        const Tensor<fp64_t> &dV, const Tensor<bool_t> &mask,
        const Tensor<fp64_t> &maxsumexp, const Tensor<fp64_t> &dst_grad,
        const Tensor<fp64_t> &tmp, const Tensor<fp64_t> &tmp_grad,
        const Tensor<fp64_t> &tmp_sumprod_slice, int redux);

} // namespace tensor
} // namespace nntile

