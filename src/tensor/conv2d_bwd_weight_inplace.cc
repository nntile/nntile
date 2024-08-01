/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/conv2d_bwd_weight_inplace.cc
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.0.0
 * */

#include "nntile/tensor/conv2d_bwd_weight_inplace.hh"
#include <algorithm>
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal_inplace.hh"
#include "nntile/starpu/conv2d_bwd_weight_inplace.hh"
#include <unistd.h>

namespace nntile::tensor
{

template <typename T>
void conv2d_bwd_weight_inplace_async(Scalar alpha, const Tensor<T> &X,
        const Tensor<T> &dY, Scalar beta, const Tensor<T> &dC,
        Index padding_m, Index padding_n)
/*! Backward 2D convolution of two tensors in WHCN format to get weight grad
 *
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * Current implementation requires dC to contain only a single tile
 *
 * The following operation is performed:
 *      `dC` = `alpha`*`f(X, dY)` + `beta`*`dC`,
 * where `f` operation does the following:
 *      `f[i,j,k,l]` = \sum_l \sum_m \sum_n `X[m,n,k,b]`
 *      * `dY[m + offset_m - i,n + offset_n - j,l,b]`
 *
 * Generally, `X` represents input of `Conv2d` layer, `dY` represents output
 * grad of `Conv2d` layer and `dC` represents weight grad of `Conv2d` layer.
 *
 * X must be of shape (W_in, H_in, C_in, N)
 * X must have basetile (W_in_tile, H_in_tile, C_in, N_tile)
 * dY must be of shape (W_out, H_out, C_out, N)
 * dY must have basetile (W_out_tile, H_out_tile, C_out, N_tile)
 * dC must be of shape (K_x, K_y, C_in, C_out)
 * dC must have basetile (K_x, K_y, C_in, C_out)
 * with W_in=W_out-K_x+2padding_m+1
 * and H_in=H_out-K_y+2padding_n+1
 *
 * @param[in] X: Input tensor, that is usually an input for Conv2d.
 * @param[in] dY: Input tensor, that is usually an output grad for Conv2d.
 * @param[inout] dC: Resulting tensor, that is usually a weight grad of Conv2d.
 * @param[in] padding_m: Padding on the second axis of the input
 * @param[in] padding_n: Padding on the first axis of the input
 * */
{
    // Check dimensions
    if(4 != X.ndim)
    {
        throw std::runtime_error("4 != X.ndim");
    }
    if(4 != dY.ndim)
    {
        throw std::runtime_error("4 != dY.ndim");
    }
    if(4 != dC.ndim)
    {
        throw std::runtime_error("4 != dC.ndim");
    }
    // Check shapes of tensors
    if(dY.shape[0] != X.shape[0] - dC.shape[0] + 1 + 2*padding_m)
    {
        throw std::runtime_error("dY.shape[0] != X.shape[0] - "
                                 "dC.shape[0] + 1 + 2*padding_n");
    }
    if(dY.shape[1] != X.shape[1] - dC.shape[1] + 1 + 2*padding_n)
    {
        throw std::runtime_error("dY.shape[1] != X.shape[1] - "
                                 "dC.shape[1] + 1 + 2*padding_m");
    }
    if(dY.shape[2] != dC.shape[3])
    {
        throw std::runtime_error("dY.shape[2] != dC.shape[3]");
    }
    if(X.shape[2] != dC.shape[2])
    {
        throw std::runtime_error("X.shape[2] != dC.shape[2]");
    }
    if(dY.shape[3] != X.shape[3])
    {
        throw std::runtime_error("dY.shape[3] != X.shape[3]");
    }
    // Check base tiles
    if(dY.basetile_shape[2] != dY.shape[2])
    {
        throw std::runtime_error("dY.basetile_shape[2] != dY.shape[2]");
    }
    if(X.basetile_shape[2] != X.shape[2])
    {
        throw std::runtime_error("X.basetile_shape[2] != X.shape[2]");
    }
    if(dY.basetile_shape[3] != X.basetile_shape[3])
    {
        throw std::runtime_error("dY.basetile_shape[3] != "
                "X.basetile_shape[3]");
    }
    if(dC.shape != dC.basetile_shape)
    {
        throw std::runtime_error("dC.shape != dC.basetile_shape");
    }

    // There is only single dC tile, so a loop over tiles of dC is omitted
    auto dC_tile_traits = dC.get_tile_traits(0);
    auto dC_tile_handle = dC.get_tile_handle(0);
    Scalar dC_tile_beta = beta;
    bool initialized = false;
    // Loop through X tiles
    for(Index i = 0; i < X.grid.nelems; ++i)
    {
        auto X_tile_index = X.grid.linear_to_index(i);
        auto X_tile_traits = X.get_tile_traits(i);
        auto X_tile_handle = X.get_tile_handle(i);
        // Get start and end coordinates of dst tile within X tensor
        Index X_start_m = X_tile_index[0] * X.basetile_shape[0];
        Index X_end_m = X_start_m + X_tile_traits.shape[0];
        Index X_start_n = X_tile_index[1] * X.basetile_shape[1];
        Index X_end_n = X_start_n + X_tile_traits.shape[1];
        // Get dY start and end coordinates that interact with X through
        // provided kernel
        Index dY_start_m = X_start_m + padding_m - dC.shape[0] + 1;
        Index dY_end_m = X_end_m + padding_m + 1;
        Index dY_start_n = X_start_n + padding_n - dC.shape[1] + 1;
        Index dY_end_n = X_end_n + padding_n + 1;
        // Get dY tile coordinates that interact with dX tile
        Index dY_start_tile_m = dY_start_m / dY.basetile_shape[0];
        Index dY_end_tile_m = (dY_end_m-1) / dY.basetile_shape[0] + 1;
        Index dY_start_tile_n = dY_start_n / dY.basetile_shape[1];
        Index dY_end_tile_n = (dY_end_n-1) / dY.basetile_shape[1] + 1;
        // Case of big padding: dX tile does not require any convolution
        if(dY_end_tile_m <= 0 or dY_start_tile_m >= dY.grid.shape[0]
                or dY_end_tile_n <= 0 or dY_start_tile_n >= dY.grid.shape[1])
        {
            // Cycle to the next X tile
            continue;
        }
        // Loop through corresponding dY tiles
        std::vector<Index> dY_tile_index(X_tile_index);
        Index start_m = std::max(dY_start_tile_m, Index(0));
        Index end_m = std::min(dY_end_tile_m, dY.grid.shape[0]);
        Index start_n = std::max(dY_start_tile_n, Index(0));
        Index end_n = std::min(dY_end_tile_n, dY.grid.shape[1]);
        for(Index dY_i = start_m; dY_i < end_m; ++dY_i)
        {
            dY_tile_index[0] = dY_i;
            for(Index dY_j = start_n; dY_j < end_n; ++dY_j)
            {
                dY_tile_index[1] = dY_j;
                auto dY_tile_traits = dY.get_tile_traits(dY_tile_index);
                auto dY_tile_handle = dY.get_tile_handle(dY_tile_index);
                Index offset_m = dY_i*dY.basetile_shape[0] - X_start_m
                    - padding_m;
                Index offset_n = dY_j*dY.basetile_shape[1] - X_start_n
                    - padding_n;
                starpu::conv2d_bwd_weight_inplace::submit<T>(
                        X_tile_traits.shape[0], X_tile_traits.shape[1],
                        X_tile_traits.shape[2], X_tile_traits.shape[3],
                        dY_tile_traits.shape[0], dY_tile_traits.shape[1],
                        dY_tile_traits.shape[2], offset_m, offset_n, alpha,
                        X_tile_handle, dY_tile_handle, dC_tile_traits.shape[0],
                        dC_tile_traits.shape[1], dC_tile_beta, dC_tile_handle);
                dC_tile_beta = 1.0;
                initialized = true;
            }
        }
    }
    if(not initialized)
    {
        // Clear if beta is 0
        if(beta == 0.0)
        {
            starpu::clear::submit(dC_tile_handle);
        }
        // Scale inplace if beta is not 1.0 or 0.0
        if(beta != 1.0)
        {
            starpu::scal_inplace::submit<T>(dC.nelems, beta, dC_tile_handle);
        }
        // Do nothing if beta is 1.0
    }
}

template <typename T>
void conv2d_bwd_weight_inplace(Scalar alpha, const Tensor<T> &X,
        const Tensor<T> &dY, Scalar beta, const Tensor<T> &dC,
        Index padding_m, Index padding_n)
/*! Blocking version of conv2d_bwd_weight_inplace_async<T>.
 *
 * @param[in] X: Input tensor, that is usually an input for Conv2d.
 * @param[in] dY: Input tensor, that is usually an output grad for Conv2d.
 * @param[inout] dC: Resulting tensor, that is usually a weight grad of Conv2d.
 * @param[in] padding_m: Padding on the second axis of the input
 * @param[in] padding_n: Padding on the first axis of the input
 * */
{
    conv2d_bwd_weight_inplace_async<T>(alpha, X, dY, beta, dY, padding_m,
            padding_n);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void conv2d_bwd_weight_inplace_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &X,
        const Tensor<bf16_t> &dY, Scalar beta, const Tensor<bf16_t> &dC,
        Index padding_m, Index padding_n);

template
void conv2d_bwd_weight_inplace_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &X,
        const Tensor<fp32_t> &dY, Scalar beta, const Tensor<fp32_t> &dC,
        Index padding_m, Index padding_n);

template
void conv2d_bwd_weight_inplace_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &X,
        const Tensor<fp32_fast_tf32_t> &dY, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dC, Index padding_m, Index padding_n);

template
void conv2d_bwd_weight_inplace_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &X,
        const Tensor<fp64_t> &dY, Scalar beta, const Tensor<fp64_t> &dC,
        Index padding_m, Index padding_n);

// Explicit instantiation of template
template
void conv2d_bwd_weight_inplace<bf16_t>(Scalar alpha, const Tensor<bf16_t> &X,
        const Tensor<bf16_t> &dY, Scalar beta, const Tensor<bf16_t> &dC,
        Index padding_m, Index padding_n);

template
void conv2d_bwd_weight_inplace<fp32_t>(Scalar alpha, const Tensor<fp32_t> &X,
        const Tensor<fp32_t> &dY, Scalar beta, const Tensor<fp32_t> &dC,
        Index padding_m, Index padding_n);

template
void conv2d_bwd_weight_inplace<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &X,
        const Tensor<fp32_fast_tf32_t> &dY, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &dC, Index padding_m, Index padding_n);

template
void conv2d_bwd_weight_inplace<fp64_t>(Scalar alpha, const Tensor<fp64_t> &X,
        const Tensor<fp64_t> &dY, Scalar beta, const Tensor<fp64_t> &dC,
        Index padding_m, Index padding_n);

} // namespace nntile::tensor
