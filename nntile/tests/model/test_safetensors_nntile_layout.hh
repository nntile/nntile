/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/test_safetensors_nntile_layout.hh
 * Map SafeTensors payload (C-order row-major) to NNTile Fortran linear layout.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/io/safetensors.hh>

namespace nntile::test::safetensors_nntile_layout
{

inline Index shape_volume(const std::vector<std::int64_t> &shape)
{
    Index vol = 1;
    for(const auto d : shape)
    {
        vol *= static_cast<Index>(d);
    }
    return vol;
}

inline Index c_order_linear_index(
    const std::vector<std::int64_t> &shape,
    const std::vector<Index> &idx)
{
    Index off = 0;
    Index stride = 1;
    for(std::size_t d = shape.size(); d-- > 0;)
    {
        off += idx[d] * stride;
        stride *= static_cast<Index>(shape[d]);
    }
    return off;
}

inline Index f_order_linear_index(
    const std::vector<std::int64_t> &shape,
    const std::vector<Index> &idx)
{
    Index off = 0;
    Index stride = 1;
    for(std::size_t d = 0; d < shape.size(); ++d)
    {
        off += idx[d] * stride;
        stride *= static_cast<Index>(shape[d]);
    }
    return off;
}

//! SafeTensors stores tensors in C-order; NNTile ``bind_data`` uses Fortran
//! linearization (first index stride 1). Convert element-wise.
template <typename T>
inline void c_safetensors_to_nntile_fortran(
    const std::uint8_t *raw,
    const std::vector<std::int64_t> &shape,
    std::vector<T> &out)
{
    if(raw == nullptr)
    {
        throw std::invalid_argument(
            "c_safetensors_to_nntile_fortran: null raw buffer");
    }
    if(shape.empty())
    {
        throw std::invalid_argument(
            "c_safetensors_to_nntile_fortran: empty shape");
    }
    const Index vol = shape_volume(shape);
    const auto expected_bytes =
        static_cast<std::size_t>(vol) * sizeof(T);
    out.resize(static_cast<std::size_t>(vol));
    if(shape.size() == 1)
    {
        std::memcpy(out.data(), raw, expected_bytes);
        return;
    }
    std::vector<Index> idx(shape.size(), 0);
    for(Index f_lin = 0; f_lin < vol; ++f_lin)
    {
        const Index c_lin = c_order_linear_index(shape, idx);
        const Index f_at = f_order_linear_index(shape, idx);
        if(f_at != f_lin)
        {
            throw std::logic_error(
                "c_safetensors_to_nntile_fortran: index walk mismatch");
        }
        out[static_cast<std::size_t>(f_lin)] =
            reinterpret_cast<const T *>(raw)[static_cast<std::size_t>(c_lin)];
        Index dim = 0;
        for(;;)
        {
            idx[static_cast<std::size_t>(dim)] += 1;
            if(idx[static_cast<std::size_t>(dim)] <
                static_cast<Index>(shape[static_cast<std::size_t>(dim)]))
            {
                break;
            }
            idx[static_cast<std::size_t>(dim)] = 0;
            if(dim + 1 >= static_cast<Index>(shape.size()))
            {
                break;
            }
            ++dim;
        }
    }
}

template <typename T>
inline void read_tensor_nntile_fortran(
    const nntile::io::SafeTensorsReader &reader,
    const char *name,
    std::vector<T> &out)
{
    const auto &info = reader.tensor_info(name);
    const auto raw = reader.read_tensor(name);
    const auto expected =
        static_cast<std::size_t>(shape_volume(info.shape)) * sizeof(T);
    if(raw.size() != expected)
    {
        throw std::runtime_error(
            std::string("read_tensor_nntile_fortran: byte size mismatch for ")
            + name);
    }
    c_safetensors_to_nntile_fortran<T>(raw.data(), info.shape, out);
}

} // namespace nntile::test::safetensors_nntile_layout
