/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/optim.hh
 * Convenience header for NNTile optimizer classes.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/optim/optimizer.hh>
#include <nntile/graph/optim/sgd.hh>
#include <nntile/graph/optim/adam.hh>
#include <nntile/graph/optim/adamw.hh>
