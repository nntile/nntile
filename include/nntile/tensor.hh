/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor.hh
 * Header for Tensor<T> class with corresponding operations
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

// Get Tensor<T> class, all headers are included explicitly
#include <nntile/base_types.hh>
#include <nntile/tensor/traits.hh>
#include <nntile/tensor/tensor.hh>

// Tensor operations
#include <nntile/tensor/randn.hh>
#include <nntile/tensor/copy.hh>
#include <nntile/tensor/gemm.hh>
#include <nntile/tensor/bias.hh>
#include <nntile/tensor/gelu.hh>

