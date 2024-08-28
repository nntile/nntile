/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/conv2d_bwd_input_inplace.cc
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of input
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/conv2d_bwd_input_inplace.hh"
#include <algorithm>
#include <array>
#include <unistd.h>
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal_inplace.hh"
#include "nntile/starpu/conv2d_bwd_input_inplace.hh"

namespace nntile::tensor
{

template <typename T>
void conv2d_bwd_input_inplace_async(Scalar alpha, const Tensor<T> &dY,
        const Tensor<T> &C, Scalar beta, const Tensor<T> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation)
/*! Backward 2D convolution of two tensors in WHCN format to get input grad
 *
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * Current implementation requires C to contain only a single tile
 *
 * The following operation is performed:
 *      `dX` = `alpha`*`f(dY, C)` + `beta`*`dX`,
 * where `f` operation does the following:
 *      `f[i,j,k,b]` = \sum_l \sum_m \sum_n `dY[m,n,l,b]`
 *      * `C[(i + offset_m - stride_m*m) / dilation_m,
 *           (j + offset_n - stride_n*n) / dilation_n,k,l]`
 * with `(i + offset_m - stride_m*m) % dilation_m == 0`
 * and `(j + offset_n - stride_n*n) % dilation_n == 0`
 *
 * Generally, `dY` represents output grad of `Conv2d` layer, `C` represents
 * kernel of `Conv2d` layer and `dX` represents input grad of `Conv2d` layer.
 *
 * dY must be of shape (W_in, H_in, C_in, N)
 * dY must have basetile (W_in_tile, H_in_tile, C_in, N_tile)
 * C must be of shape (K_x, K_y, C_out, C_in)
 * C must have basetile (K_x, K_y, C_out, C_in)
 * dX must be of shape (W_out, H_out, C_out, N)
 * with W_in=(W_out+2*padding[0]-dilation[0]*(K_x-1)-1)/stride[0]+1
 * and H_in=(H_out+2*padding[1]-dilation[1]*(K_y-1)-1)/stride[1]+1
 * dX must have basetile (W_out_tile, H_out_tile, C_out, N_tile)
 *
 * @param[in] dY: Input tensor, that is usually an output grad for Conv2d.
 * @param[in] C: Input tensor, that is usually a kernel for Conv2d.
 * @param[inout] dX: Resulting tensor, that is usually an input grad of Conv2d.
 * @param[in] padding: Padding of the convolution
 * @param[in] stride: Stride of the convolution
 * @param[in] dilation: Padding of the convolution
 * */
{
    // Check dimensions
    if(4 != dY.ndim)
    {
        throw std::runtime_error("4 != dY.ndim");
    }
    if(4 != C.ndim)
    {
        throw std::runtime_error("4 != C.ndim");
    }
    if(4 != dX.ndim)
    {
        throw std::runtime_error("4 != dX.ndim");
    }
    // Check shapes of tensors
    if(dY.shape[0] != (dX.shape[0]+2*padding[0]-dilation[0]*(C.shape[0]-1)-1)
            / stride[0] + 1)
    {
        throw std::runtime_error("Incorrect dY.shape[0]");
    }
    if(dY.shape[1] != (dX.shape[1]+2*padding[1]-dilation[1]*(C.shape[1]-1)-1)
            / stride[1] + 1)
    {
        throw std::runtime_error("Incorrect dY.shape[1]");
    }
    if(dY.shape[2] != C.shape[3])
    {
        throw std::runtime_error("dY.shape[2] != C.shape[3]");
    }
    if(dX.shape[2] != C.shape[2])
    {
        throw std::runtime_error("dX.shape[2] != C.shape[2]");
    }
    if(dY.shape[3] != dX.shape[3])
    {
        throw std::runtime_error("dY.shape[3] != dX.shape[3]");
    }
    // Check base tiles
    if(dY.basetile_shape[2] != dY.shape[2])
    {
        throw std::runtime_error("dY.basetile_shape[2] != dY.shape[2]");
    }
    if(dX.basetile_shape[2] != dX.shape[2])
    {
        throw std::runtime_error("dX.basetile_shape[2] != dX.shape[2]");
    }
    if(dY.basetile_shape[3] != dX.basetile_shape[3])
    {
        throw std::runtime_error("dY.basetile_shape[3] != "
                "dX.basetile_shape[3]");
    }
    if(C.shape != C.basetile_shape)
    {
        throw std::runtime_error("C.shape != C.basetile_shape");
    }

    // Loop through output tiles
    for(Index i = 0; i < dX.grid.nelems; ++i)
    {
        auto dX_tile_index = dX.grid.linear_to_index(i);
        auto dX_tile_traits = dX.get_tile_traits(i);
        auto dX_tile_handle = dX.get_tile_handle(i);
        // Get start and end coordinates of dst tile within dX tensor
        Index dX_start_m = dX_tile_index[0] * dX.basetile_shape[0];
        Index dX_end_m = dX_start_m + dX_tile_traits.shape[0];
        Index dX_start_n = dX_tile_index[1] * dX.basetile_shape[1];
        Index dX_end_n = dX_start_n + dX_tile_traits.shape[1];
        // Get start and end indices `m` and `n` from the operation:
        //      `f[i,j,k,b]` = \sum_l \sum_m \sum_n `dY[m,n,l,b]`
        //      * `C[(i + offset_m - stride_m*m) / dilation_m,
        //           (j + offset_n - stride_n*n) / dilation_n,k,l]`
        // Limits are `0 <= i+offset_m-stride_m*m <= dilation_m*(C.shape[0]-1)`
        // Therefore,
        //      `m >= ceil((i+offset_m-dilation_m*(C.shape[0]-1)) / stride_m)`
        //      `m <= floor((i+offset_m) / stride_m)`.
        Index dY_start_m = (dX_start_m+padding[0]-dilation[0]*(C.shape[0]-1)
                +stride[0]-1) / stride[0];
        Index dY_end_m = (dX_end_m-1+padding[0]+stride[0]) / stride[0];
        Index dY_start_n = (dX_start_n+padding[1]-dilation[1]*(C.shape[1]-1)
                +stride[1]-1) / stride[1];
        Index dY_end_n = (dX_end_n-1+padding[1]+stride[1]) / stride[1];
        // Get dY tile coordinates that interact with dX tile
        Index dY_start_tile_m = dY_start_m / dY.basetile_shape[0];
        Index dY_end_tile_m = (dY_end_m-1) / dY.basetile_shape[0] + 1;
        Index dY_start_tile_n = dY_start_n / dY.basetile_shape[1];
        Index dY_end_tile_n = (dY_end_n-1) / dY.basetile_shape[1] + 1;
        // Case of big padding: dX tile does not require any convolution
        if(dY_end_tile_m <= 0 or dY_start_tile_m >= dY.grid.shape[0]
                or dY_end_tile_n <= 0 or dY_start_tile_n >= dY.grid.shape[1])
        {
            // Clear if beta is zero
            if(beta == 0.0)
            {
                starpu::clear::submit(dX_tile_handle);
            }
            // Scale inplace if beta is neither zero nor one
            else if(beta != 1.0)
            {
                starpu::scal_inplace::submit<T>(dX_tile_traits.nelems, beta,
                        dX_tile_handle);
            }
            // Do nothing if beta is one
            // Cycle to the next dX tile
            continue;
        }
        // Loop through corresponding dY tiles
        std::vector<Index> dY_tile_index(dX_tile_index);
        Index start_m = std::max(dY_start_tile_m, Index(0));
        Index end_m = std::min(dY_end_tile_m, dY.grid.shape[0]);
        Index start_n = std::max(dY_start_tile_n, Index(0));
        Index end_n = std::min(dY_end_tile_n, dY.grid.shape[1]);
        Scalar dX_tile_beta = beta;
        for(Index dY_i = start_m; dY_i < end_m; ++dY_i)
        {
            dY_tile_index[0] = dY_i;
            for(Index dY_j = start_n; dY_j < end_n; ++dY_j)
            {
                dY_tile_index[1] = dY_j;
                auto dY_tile_traits = dY.get_tile_traits(dY_tile_index);
                auto dY_tile_handle = dY.get_tile_handle(dY_tile_index);
                Index offset_m = dX_start_m + padding[0]
                    - stride[0]*dY_i*dY.basetile_shape[0];
                Index offset_n = dX_start_n + padding[1]
                    - stride[1]*dY_j*dY.basetile_shape[1];
                starpu::conv2d_bwd_input_inplace::submit<T>(
                        dY_tile_traits.shape[0], dY_tile_traits.shape[1],
                        stride[0], stride[1], dY_tile_traits.shape[2],
                        dY_tile_traits.shape[3], C.shape[0], C.shape[1],
                        dilation[0], dilation[1], dX_tile_traits.shape[2],
                        offset_m, offset_n, alpha, dY_tile_handle,
                        C.get_tile_handle(0), dX_tile_traits.shape[0],
                        dX_tile_traits.shape[1], dX_tile_beta, dX_tile_handle);
                dX_tile_beta = 1.0;
            }
        }
    }
}

template <typename T>
void conv2d_bwd_input_inplace(Scalar alpha, const Tensor<T> &dY,
        const Tensor<T> &C, Scalar beta, const Tensor<T> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation)
/*! Blocking version of conv2d_bwd_input_inplace_async<T>.
 *
 * @param[in] dY: Input tensor, that is usually an output grad for Conv2d.
 * @param[in] C: Input tensor, that is usually a kernel for Conv2d.
 * @param[inout] dX: Resulting tensor, that is usually an input grad of Conv2d.
 * @param[in] padding: Padding of the convolution
 * @param[in] stride: Stride of the convolution
 * @param[in] dilation: Padding of the convolution
 * */
{
    conv2d_bwd_input_inplace_async<T>(alpha, dY, C, beta, dX, padding, stride,
            dilation);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void conv2d_bwd_input_inplace_async<bf16_t>(Scalar alpha,
        const Tensor<bf16_t> &dY, const Tensor<bf16_t> &C, Scalar beta,
        const Tensor<bf16_t> &dX, std::array<Index, 2> padding,
        std::array<Index, 2> stride, std::array<Index, 2> dilation);

template
void conv2d_bwd_input_inplace_async<fp32_t>(Scalar alpha,
        const Tensor<fp32_t> &dY, const Tensor<fp32_t> &C, Scalar beta,
        const Tensor<fp32_t> &dX, std::array<Index, 2> padding,
        std::array<Index, 2> stride, std::array<Index, 2> dilation);

template
void conv2d_bwd_input_inplace_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &dY, const Tensor<fp32_fast_tf32_t> &C,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

template
void conv2d_bwd_input_inplace_async<fp64_t>(Scalar alpha,
        const Tensor<fp64_t> &dY, const Tensor<fp64_t> &C, Scalar beta,
        const Tensor<fp64_t> &dX, std::array<Index, 2> padding,
        std::array<Index, 2> stride, std::array<Index, 2> dilation);

// Explicit instantiation of template
template
void conv2d_bwd_input_inplace<bf16_t>(Scalar alpha, const Tensor<bf16_t> &dY,
        const Tensor<bf16_t> &C, Scalar beta, const Tensor<bf16_t> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

template
void conv2d_bwd_input_inplace<fp32_t>(Scalar alpha, const Tensor<fp32_t> &dY,
        const Tensor<fp32_t> &C, Scalar beta, const Tensor<fp32_t> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

template
void conv2d_bwd_input_inplace<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &dY, const Tensor<fp32_fast_tf32_t> &C,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

template
void conv2d_bwd_input_inplace<fp64_t>(Scalar alpha, const Tensor<fp64_t> &dY,
        const Tensor<fp64_t> &C, Scalar beta, const Tensor<fp64_t> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

} // namespace nntile::tensor
