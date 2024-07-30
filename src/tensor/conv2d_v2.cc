/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/conv2d.cc
 * Tensor wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#include "nntile/tensor/conv2d_v2.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/conv2d.hh"
#include <unistd.h>

namespace nntile::tensor
{

template <typename T>
void conv2d_v2_inplace_async(Scalar alpha, const Tensor<T> &src, const Tensor<T> &kernel,
                  Scalar beta, const Tensor<T> &dst, Index padding_m,
                  Index padding_n)
//! Tensor<T> 2D-Convolution between 2 matrices
/*! Reshapes input tensors into 2-dimensional arrays
 * and performs the 2D-Convolution
 * src must be of shape (W_in, H_in, C_in, N)
 * src must have basetile (W_in_tile, H_in_tile, C_in, N_tile)
 * kernel must be of shape (K_x, K_y, C_in, C_out)
 * kernel must have basetile (K_x, K_y, C_in, C_out)
 * dst must be of shape (W_out, H_out, C_out, N)
 * with W_out=W_in-K_x+2paddint_m+1
 * and H_out=H_in-K_y+2padding_n+1
 * dst must have basetile (W_out_tile, H_out_tile, C_out, N_tile)
 *
 * @param[in] src: Input tensor, that is reshaped into 2D array
 * @param[in] kernel: Input tensor, that is reshaped into 2D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 2D array
 * @param[in] padding_m: Padding on the second axis of the input
 * @param[in] padding_n: Padding on the first axis of the input
 * */
{
    // Check dimensions
    if(4 != src.ndim)
    {
        throw std::runtime_error("4 != src.ndim");
    }
    if(4 != dst.ndim)
    {
        throw std::runtime_error("4 != dst.ndim");
    }
    // Check shapes of tensors
    if(dst.shape[0] != src.shape[0] - kernel.shape[0] + 1 + 2*padding_m)
    {
        throw std::runtime_error("dst.shape[0] != src.shape[0] - "
                                 "kernel.shape[0] + 1 + 2*padding_n");
    }
    if(dst.shape[1] != src.shape[1] - kernel.shape[1] + 1 + 2*padding_n)
    {
        throw std::runtime_error("dst.shape[1] != src.shape[1] - "
                                 "kernel.shape[1] + 1 + 2*padding_m");
    }
    if(src.shape[2] != kernel.shape[2])
    {
        throw std::runtime_error("src.shape[2] != kernel.shape[2]")
    }
    if(dst.shape[2] != kernel.shape[3])
    {
        throw std::runtime_error("dst.shape[2] != kernel.shape[3]")
    }
    if(src.shape[3] != dst.shape[3])
    {
        throw std::runtime_error("src.shape[3] != dst.shape[3]")
    }
    // Check base tiles
    if(src.basetile_shape[2] != src.shape[2])
    {
        throw std::runtime_error("src.basetile_shape[2] != src.shape[2]")
    }
    if(dst.basetile_shape[2] != dst.shape[2])
    {
        throw std::runtime_error("dst.basetile_shape[2] != dst.shape[2]")
    }
    if(src.basetile_shape[3] != dst.basetile_shape[3])
    {
        throw std::runtime_error("src.basetile_shape[3] != "
                "dst.basetile_shape[3]")
    }
    if(kernel.shape != kernel.basetile_shape)
    {
        throw std::runtime_error("kernel.shape != kernel.basetile_shape");
    }

    // Loop through output tiles
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_index = dst.grid.linear_to_index(i);
        auto dst_tile_traits = dst.get_tile_traits(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        // Get start and end coordinates of dst tile within dst tensor
        Index dst_start_m = dst_tile_index[0] * dst.basetile_shape[0];
        Index dst_end_m = dst_start_m + dst_tile_traits.shape[0];
        Index dst_start_n = dst_tile_index[1] * dst.basetile_shape[1];
        Index dst_end_n = dst_start_n + dst_tile_traits.shape[1];
        // Get src start and end coordinates that interact with dst through
        // provided kernel
        Index src_start_m = dst_start_m - padding_m;
        Index src_end_m = dst_end_m - padding_m + kernel.shape[0];
        Index src_start_n = dst_start_n - padding_n;
        Index src_end_n = dst_end_n - padding_n + kernel.shape[1];
        // Get src tile coordinates that interact with dst tile
        Index src_start_tile_m = src_start_m / src.basetile_shape[0];
        Index src_end_tile_m = (src_end_m-1) / src.basetile_shape[0] + 1;
        Index src_start_tile_n = src_start_n / src.basetile_shape[0];
        Index src_end_tile_n = (src_end_n-1) / src.basetile_shape[0] + 1;
        // Case of big padding: dst tile does not require any convolution
        if(src_end_tile_m <= 0 or src_start_tile_m >= src.grid.shape[0]
                or src_end_tile_n <= 0 or src_start_tile_n >= src.grid.shape[1])
        {
            // Clear if beta is zero
            if(beta == 0.0)
            {
                starpu::clear::submit(dst_tile_handle);
            }
            // Scale inplace if beta is neither zero nor one
            else if(beta != 1.0)
            {
                starpu::scal_inplace<T>(beta, dst_tile_handle);
            }
            // Do nothing if beta is one
            // Cycle to the next dst tile
            continue;
        }
        // Loop through corresponding src tiles
        std::vector<Index> src_tile_index(dst_tile_index);
        Index start_m = std::max(src_start_tile_m, 0);
        Index end_m = std::min(src_end_tile_m, src.grid.shape[0]);
        Index start_n = std::max(src_start_tile_n, 0);
        Index end_n = std::min(src_end_tile_n, src.grid.shape[1]);
        for(Index src_i = start_m; src_i < end_m; ++src_i)
        {
            src_tile_index[0] = src_i;
            for(Index src_j = start_n; src_j < end_n; ++src_j)
            {
                src_tile_index[1] = src_j;
                auto src_tile_traits = src.get_tile_traits(src_tile_index);
                auto src_tile_handle = src.get_tile_handle(src_tile_index);
                Index offset_m = src_i*src.basetile_shape[0] - dst_start_m
                    + padding_m;
                Index offset_n = src_j*src.basetile_shape[1] - dst_start_n
                    + padding_n;
                starpu::conv2d_v2::submit<T>(src_tile_traits.shape[0],
                        src_tile_traits.shape[1], src_tile_traits.shape[2],
                        src_tile_traits.shape[3], offset_m, offset_n, alpha,
                        src_tile_handle, kernel.shape[0], kernel.shape[1],
                        kernel.shape[3], kernel.get_tile_handle(0), beta,
                        dst_tile_handle);
            }
        }
    }
}

template <typename T>
void conv2d_v2_inplace(Scalar alpha, const Tensor<T> &src,
        const Tensor<T> &kernel, Scalar beta, const Tensor<T> &dst,
        Index padding_m, Index padding_n)
//! Tensor<T> 2D-Convolution between 2 matrices
/*! Blocking version of conv2d_async<T>.
 * Reshapes input tensors into 2-dimensional arrays
 * and performs the 2D-Convolution
 *
 * @param[in] src: Input tensor, that is reshaped into 2D array
 * @param[in] kernel: Input tensor, that is reshaped into 2D array
 * @param[out] dst: Resulting tensor, that is reshaped into 2D array
 * @param[in] padding_m: Padding on the second axis of the input
 * @param[in] padding_n: Padding on the first axis of the input
 * */
{
    conv2d_v2_inplace_async<T>(alpha, src, kernel, beta, dst, padding_m,
            padding_n);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template void conv2d_v2_inplace_async<fp32_t>(Scalar alpha,
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &kernel,
        Scalar beta, const Tensor<fp32_t> &dst, Index padding_m,
        Index padding_n);

template void conv2d_v2_inplace_async<fp64_t>(Scalar alpha,
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &kernel,
        Scalar beta, const Tensor<fp64_t> &dst, Index padding_m,
        Index padding_n);

// Explicit instantiation of template
template void conv2d_v2_inplace<fp32_t>(Scalar alpha,
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &kernel,
        Scalar beta, const Tensor<fp32_t> &dst, Index padding_m,
        Index padding_n);

template void conv2d_v2_inplace<fp64_t>(Scalar alpha,
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &kernel,
        Scalar beta, const Tensor<fp64_t> &dst, Index padding_m,
        Index padding_n);

} // namespace nntile::tensor
