/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/graph_ops.hh
 * TileGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/add.hh>
#include <nntile/graph/tile/add_inplace.hh>
#include <nntile/graph/tile/clear.hh>
#include <nntile/graph/tile/fill.hh>
#include <nntile/graph/tile/gemm.hh>
#include <nntile/graph/tile/gelu.hh>
#include <nntile/graph/tile/gelu_backward.hh>
#include <nntile/graph/tile/gelu_inplace.hh>
#include <nntile/graph/tile/gelutanh.hh>
#include <nntile/graph/tile/gelutanh_backward.hh>
#include <nntile/graph/tile/gelutanh_inplace.hh>
#include <nntile/graph/tile/multiply.hh>
#include <nntile/graph/tile/adam_step.hh>
#include <nntile/graph/tile/adamw_step.hh>
#include <nntile/graph/tile/sgd_step.hh>
#include <nntile/graph/tile/pow.hh>
#include <nntile/graph/tile/relu.hh>
#include <nntile/graph/tile/relu_backward.hh>
#include <nntile/graph/tile/silu.hh>
#include <nntile/graph/tile/silu_backward.hh>
#include <nntile/graph/tile/silu_inplace.hh>
#include <nntile/graph/tile/sqrt.hh>

