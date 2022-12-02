/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel.hh
 * General info about namespace nntile::kernel
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-02
 * */

#pragma once

#include <nntile/kernel/bias.hh>
#include <nntile/kernel/gelu.hh>
#include <nntile/kernel/gelutanh.hh>
#include <nntile/kernel/dgelu.hh>
#include <nntile/kernel/dgelutanh.hh>
#include <nntile/kernel/hypot.hh>
#include <nntile/kernel/normalize.hh>
#include <nntile/kernel/prod.hh>
#include <nntile/kernel/randn.hh>
#include <nntile/kernel/relu.hh>
#include <nntile/kernel/subcopy.hh>
#include <nntile/kernel/sumnorm.hh>

namespace nntile
{
//! @namespace nntile::kernel
/*! This namespace holds low-level routines for codelets
 * */
namespace kernel
{

} // namespace kernel
} // namespace nntile

