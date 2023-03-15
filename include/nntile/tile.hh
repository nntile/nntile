/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile.hh
 * Header for Tile<T> class with corresponding operations
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-02-14
 * */

#pragma once

// Get Tile<T> class
#include <nntile/tile/tile.hh>

// Tile<T> operations
#include <nntile/tile/axpy.hh>
#include <nntile/tile/bias.hh>
#include <nntile/tile/clear.hh>
#include <nntile/tile/copy.hh>
#include <nntile/tile/copy_intersection.hh>
#include <nntile/tile/gelu.hh>
#include <nntile/tile/gelutanh.hh>
#include <nntile/tile/dgelu.hh>
#include <nntile/tile/dgelutanh.hh>
#include <nntile/tile/drelu.hh>
#include <nntile/tile/gemm.hh>
#include <nntile/tile/nrm2.hh>
#include <nntile/tile/normalize.hh>
#include <nntile/tile/prod.hh>
#include <nntile/tile/randn.hh>
#include <nntile/tile/relu.hh>
#include <nntile/tile/sumnorm.hh>
#include <nntile/tile/maxsumexp.hh>
#include <nntile/tile/softmax.hh>
#include <nntile/tile/sqrt.hh>
#include <nntile/tile/maximum.hh>
#include <nntile/tile/addcdiv.hh>
#include <nntile/tile/logsumexp.hh>

namespace nntile
{
//! @namespace nntile::tile
/*! This namespace holds high-level routines for Tile<T>
 * */
namespace tile
{

} // namespace tile
} // namespace nntile

