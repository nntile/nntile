/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/mixer.hh
 * Mixer layer
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor.hh>

namespace nntile
{

template<typename T>
class Mixer
{
    //! The first fully connected layer of the first MLP
    Tensor<T> mlp1_weight1;
    //! The second fully connected layer of the first MLP
    Tensor<T> mlp1_weight2;
    //! Temporary for GeLU of the first MLP
    Tensor<T> mlp1_gelu;
    //! The first fully connected layer of the second MLP
    Tensor<T> mlp2_weight1;
    //! The second fully connected layer of the second MLP
    Tensor<T> mlp2_weight2;
    //! Temporary for GeLU of the second MLP
    Tensor<T> mlp2_gelu;
    //! The first normalization
    Tensor<T> norm1;
    //! The second normalization
    Tensor<T> norm2;
    //! Temporary sum and sum of squares for the first normalization
    Tensor<T> sum_ssq1;
    //! Temporary sum and sum of squares for the second normalization
    Tensor<T> sum_ssq2;
    //! Temporary avg and deviation for the first normalization
    Tensor<T> avg_dev1;
    //! Temporary avg and deviation for the second normalization
    Tensor<T> avg_dev2;
    //! Regularization for the normalization
    T eps;
public:
    //! n_p: number of patches
    // n_c: number of channels
    // n_b: size of batch
    Mixer(Index n_p, Index n_p_tile, Index n_p_mlp, Index n_p_mlp_tile,
            Index n_c, Index n_c_tile, Index n_c_mlp, Index n_c_mlp_tile,
            Index n_b, Index n_b_tile, T eps_):
        mlp1_weight1({n_p_mlp, n_p}, {n_p_mlp_tile, n_p_tile}),
        mlp1_weight2({n_p, n_p_mlp}, {n_p_tile, n_p_mlp_tile}),
        mlp1_gelu({n_p_mlp, n_b, n_c}, {n_p_mlp_tile, n_b_tile, n_c_tile}),
        mlp2_weight1({n_c, n_c_mlp}, {n_c_tile, n_c_mlp_tile}),
        mlp2_weight2({n_c_mlp, n_c}, {n_c_mlp_tile, n_c_tile}),
        mlp2_gelu({n_p, n_b, n_c_mlp}, {n_p_tile, n_b_tile, n_c_mlp_tile}),
        norm1({n_p, n_b, n_c}, {n_p_tile, n_b_tile, n_c_tile}),
        norm2({n_p, n_b, n_c}, {n_p_tile, n_b_tile, n_c_tile}),
        sum_ssq1({3, n_p, n_b}, {3, n_p_tile, n_b_tile}),
        sum_ssq2({3, n_b, n_c}, {3, n_b_tile, n_c_tile}),
        avg_dev1({2, n_p, n_b}, {2, n_p_tile, n_b_tile}),
        avg_dev2({2, n_b, n_c}, {2, n_b_tile, n_c_tile}),
        eps(eps_)
    {
    }
    void init(const Tensor<T> &mlp1_weight1_, const Tensor<T> &mlp1_weight2_,
            const Tensor<T> &mlp2_weight1_, const Tensor<T> &mlp2_weight2_)
    {
        copy_intersection_async(mlp1_weight1_, mlp1_weight1);
        copy_intersection_async(mlp1_weight2_, mlp1_weight2);
        copy_intersection_async(mlp2_weight1_, mlp2_weight1);
        copy_intersection_async(mlp2_weight2_, mlp2_weight2);
        starpu_task_wait_for_all();
    }
    void forward_async(const Tensor<T> &input, const Tensor<T> &output)
    {
        if(input.ndim != 3)
        {
            throw std::runtime_error("input.ndim != 3");
        }
        if(output.ndim != 3)
        {
            throw std::runtime_error("output.ndim != 3");
        }
        if(input.shape != norm1.shape)
        {
            throw std::runtime_error("input.shape != norm1.shape");
        }
        if(input.basetile_shape != norm1.basetile_shape)
        {
            throw std::runtime_error("input.basetile_shape != "
                    "norm1.basetile_shape");
        }
        if(output.shape != input.shape)
        {
            throw std::runtime_error("output.shape != input.shape");
        }
        if(output.basetile_shape != input.basetile_shape)
        {
            throw std::runtime_error("output.basetile_shape != "
                    "input.basetile_shape");
        }
        // Skip connection
        copy_intersection_async(input, output);
        // Layer normalization over channels
        copy_intersection_async(input, norm1);
        norm_sum_ssq_async(norm1, sum_ssq1, 2);
        norm_avg_dev_async(sum_ssq1, avg_dev1, norm1.shape[2], eps);
        bias_avg_dev_async(avg_dev1, norm1, 2);
        // Apply the first fully connected layer of the first MLP
        gemm_async(T{1}, TransOp::NoTrans, mlp1_weight1, TransOp::NoTrans,
                norm1, T{0}, mlp1_gelu);
        // GeLU is too slow, use ReLU instead
        relu_async(mlp1_gelu);
        // Apply the second fully connected layer of the first MLP
        gemm_async(T{1}, TransOp::NoTrans, mlp1_weight2, TransOp::NoTrans,
                mlp1_gelu, T{1}, output);
        // Layer normalization over patches, skip connection is already inplace
        copy_intersection_async(output, norm2);
        norm_sum_ssq_async(norm2, sum_ssq2, 0);
        norm_avg_dev_async(sum_ssq2, avg_dev2, norm2.shape[0], eps);
        bias_avg_dev_async(avg_dev2, norm2, 0);
        // Apply the first fully connected layer of the second MLP
        gemm_async(T{1}, TransOp::NoTrans, norm2, TransOp::NoTrans,
                mlp2_weight1, T{0}, mlp2_gelu);
        // GeLU is too slow, use ReLU instead
        relu_async(mlp2_gelu);
        // Apply the second fully connected layer of the second MLP
        gemm_async(T{1}, TransOp::NoTrans, mlp2_gelu, TransOp::NoTrans,
                mlp2_weight2, T{1}, output);
    }
    const Tensor<T> &get_mlp1_weight1() const
    {
        return mlp1_weight1;
    }
    const Tensor<T> &get_mlp1_weight2() const
    {
        return mlp1_weight2;
    }
    const Tensor<T> &get_mlp2_weight1() const
    {
        return mlp2_weight1;
    }
    const Tensor<T> &get_mlp2_weight2() const
    {
        return mlp2_weight2;
    }
    void unregister()
    {
        mlp1_weight1.unregister();
        mlp1_weight2.unregister();
        mlp1_gelu.unregister();
        mlp2_weight1.unregister();
        mlp2_weight2.unregister();
        mlp2_gelu.unregister();
        norm1.unregister();
        norm2.unregister();
        sum_ssq1.unregister();
        sum_ssq2.unregister();
        avg_dev1.unregister();
        avg_dev2.unregister();
    }
};

} // namespace nntile
