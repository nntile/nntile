/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/pack_args.hh
 * Unpack STARPU_VALUE-style cl_args without including starpu.h.
 */

#pragma once

#include <cstddef>

namespace nntile::haul
{

inline void unpack_args_ptr_single_arg(void const *, int)
{
}

template<typename T, typename... Ts>
void unpack_args_ptr_single_arg(
    void const *cl_args,
    int nargs,
    T const *&ptr,
    Ts const *&...args)
{
    if (nargs == 0)
    {
        return;
    }
    std::size_t const arg_size =
        *reinterpret_cast<std::size_t const *>(cl_args);
    char const *char_ptr =
        reinterpret_cast<char const *>(cl_args) + sizeof(std::size_t);
    ptr = reinterpret_cast<T const *>(char_ptr);
    unpack_args_ptr_single_arg(char_ptr + arg_size, nargs - 1, args...);
}

template<typename... Ts>
void unpack_args_ptr(void const *cl_args, Ts const *&...args)
{
    int const nargs = *reinterpret_cast<int const *>(cl_args);
    char const *rest =
        reinterpret_cast<char const *>(cl_args) + sizeof(int);
    if (nargs > 0)
    {
        unpack_args_ptr_single_arg(rest, nargs, args...);
    }
}

} // namespace nntile::haul
