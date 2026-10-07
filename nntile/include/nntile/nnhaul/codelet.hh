/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/codelet.hh
 * NNHaul codelet wrapper and ``nnhaul::insert`` helpers.
 */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/data_access.hh>
#include <nntile/defs.h>

#include <nnhaul/nnhaul.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nntile::haul
{

inline std::uint64_t fnv1a(
    void const *data,
    std::size_t size,
    std::uint64_t hash) noexcept
{
    unsigned char const *bytes =
        static_cast<unsigned char const *>(data);
    for (std::size_t i = 0; i < size; ++i)
    {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

template<typename T>
T *buf_as(void **buffers, int index) noexcept
{
    return static_cast<::nnhaul::Buffer *>(buffers[index])->get_ptr<T>();
}

inline ::nnhaul::Access access_from_mode(int mode)
{
    if (mode == STARPU_REDUX)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    if (mode == STARPU_R)
    {
        return ::nnhaul::Access::Read;
    }
    if (mode == STARPU_W)
    {
        return ::nnhaul::Access::Write;
    }
    if (mode == STARPU_RW)
    {
        return ::nnhaul::Access::ReadWrite;
    }
    throw std::runtime_error("unsupported data access mode");
}

struct BufSpec
{
    int mode = 0;
    ::nnhaul::Handle *handle = nullptr;
};

struct ValSpec
{
    void const *ptr = nullptr;
    std::size_t size = 0;
};

inline void insert_task(
    ::nnhaul::Codelet &codelet,
    int worker,
    std::vector<BufSpec> const &bufs,
    void const *cl_args,
    std::size_t cl_arg_size)
{
    std::vector<::nnhaul::BufferUse> uses;
    uses.reserve(bufs.size());
    for (BufSpec const &buf : bufs)
    {
        uses.push_back(
            ::nnhaul::BufferUse{buf.handle, access_from_mode(buf.mode)});
    }
    ::nnhaul::WorkerId const id = worker < 0
        ? ::nnhaul::worker_unspecified
        : worker;
    switch (uses.size())
    {
    case 0:
        ::nnhaul::insert(codelet, {}, cl_args, cl_arg_size, id);
        break;
    case 1:
        ::nnhaul::insert(
            codelet, {uses[0]}, cl_args, cl_arg_size, id);
        break;
    case 2:
        ::nnhaul::insert(
            codelet, {uses[0], uses[1]}, cl_args, cl_arg_size, id);
        break;
    case 3:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 4:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 5:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 6:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4], uses[5]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 7:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4], uses[5],
                uses[6]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 8:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4], uses[5],
                uses[6], uses[7]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 9:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4], uses[5],
                uses[6], uses[7], uses[8]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 10:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4], uses[5],
                uses[6], uses[7], uses[8], uses[9]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 11:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4], uses[5],
                uses[6], uses[7], uses[8], uses[9], uses[10]},
            cl_args,
            cl_arg_size,
            id);
        break;
    case 12:
        ::nnhaul::insert(
            codelet,
            {uses[0], uses[1], uses[2], uses[3], uses[4], uses[5],
                uses[6], uses[7], uses[8], uses[9], uses[10], uses[11]},
            cl_args,
            cl_arg_size,
            id);
        break;
    default:
        throw std::runtime_error(
            "too many NNHaul buffers: " + std::to_string(uses.size()));
    }
}

inline void insert_values(
    ::nnhaul::Codelet &codelet,
    int worker,
    std::vector<ValSpec> const &vals,
    std::vector<BufSpec> const &bufs)
{
    std::vector<unsigned char> packed;
    int const nargs = static_cast<int>(vals.size());
    packed.resize(sizeof(int));
    std::memcpy(packed.data(), &nargs, sizeof(int));
    for (ValSpec const &val : vals)
    {
        std::size_t const at = packed.size();
        packed.resize(at + sizeof(std::size_t) + val.size);
        std::memcpy(packed.data() + at, &val.size, sizeof(std::size_t));
        if (val.size > 0)
        {
            std::memcpy(
                packed.data() + at + sizeof(std::size_t),
                val.ptr,
                val.size);
        }
    }
    insert_task(
        codelet,
        worker,
        bufs,
        packed.empty() ? nullptr : packed.data(),
        packed.size());
}

class Codelet
{
public:
    ::nnhaul::Codelet raw;

    Codelet(
        std::string const &name,
        ::nnhaul::Codelet::Kernel cpu,
        ::nnhaul::Codelet::Kernel cuda,
        ::nnhaul::Codelet::FootprintFn footprint):
        raw(name, cpu, cuda, footprint)
    {
    }

    Codelet &restrict_where(std::uint32_t)
    {
        return *this;
    }

    Codelet &restore_where()
    {
        return *this;
    }

    Codelet &set_modes_variable()
    {
        return *this;
    }

    template<typename Mode, std::size_t N>
    Codelet &set_modes_fixed(std::array<Mode, N> const &)
    {
        return *this;
    }
};

template<typename... Ts>
class CodeletTyped: public Codelet
{
public:
    static std::string get_name(std::string const &base_name)
    {
        return base_name + "_" + ::nntile::type_postfix<Ts...>();
    }

    CodeletTyped(
        std::string const &base_name,
        ::nnhaul::Codelet::Kernel cpu,
        ::nnhaul::Codelet::Kernel cuda,
        ::nnhaul::Codelet::FootprintFn footprint):
        Codelet(get_name(base_name), cpu, cuda, footprint)
    {
    }
};

template<template<typename> typename Operation, typename... Ts>
class OperationPack: public Operation<Ts>...
{
public:
    OperationPack():
        Operation<Ts>()...
    {
    }

    OperationPack &restrict_where(std::uint32_t where)
    {
        (static_cast<Operation<Ts> &>(*this).codelet.restrict_where(where),
            ...);
        return *this;
    }

    OperationPack &restore_where()
    {
        (static_cast<Operation<Ts> &>(*this).codelet.restore_where(), ...);
        return *this;
    }

    template<typename T, typename... Args>
    void submit(Args &&...args)
    {
        static_cast<Operation<T> &>(*this).submit(
            std::forward<Args>(args)...);
    }
};

} // namespace nntile::haul
