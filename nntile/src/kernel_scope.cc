/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/kernel_scope.cc
 * Build-time kernel scope gate (see include/nntile/kernel_scope.hh).
 *
 * SPDX-License-Identifier: MIT
 *
 * @version 1.0.0
 * */

#include "nntile/kernel_scope.hh"

#include "nntile/defs.h"

#include <stdexcept>
#include <string>

namespace nntile
{

namespace
{

#ifndef NNTILE_DISABLED_KERNELS
#define NNTILE_DISABLED_KERNELS ""
#endif

//! Comma/semicolon separated disable list, expanded at configure time.
char const *const kDisabled = NNTILE_DISABLED_KERNELS;

bool matches(std::string_view list, std::string_view name) noexcept
{
    std::size_t begin = 0;
    while(begin < list.size())
    {
        while(begin < list.size()
            && (list[begin] == ';' || list[begin] == ','))
        {
            ++begin;
        }
        std::size_t end = begin;
        while(end < list.size() && list[end] != ';' && list[end] != ',')
        {
            ++end;
        }
        if(list.substr(begin, end - begin) == name)
        {
            return true;
        }
        begin = end;
    }
    return false;
}

} // namespace

bool kernel_family_disabled(std::string_view name) noexcept
{
    return matches(kDisabled, name);
}

void require_kernel_family(std::string_view name)
{
    if(kernel_family_disabled(name))
    {
        throw std::runtime_error(
            "nntile: kernel family '" + std::string(name)
            + "' is disabled in this transformers-only build "
              "(NNTILE_DISABLED_KERNELS)");
    }
}

} // namespace nntile
