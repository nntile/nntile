/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/model/test_safetensors_nntile_layout.hh
 * Map SafeTensors payload (graph) to NNTile bind_data layout.
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

//! SafeTensors and NNTile both store tensors in graph; copy bytes directly.
template <typename T>
inline void copy_safetensors_to_nntile_layout(
    const std::uint8_t *raw,
    const std::vector<std::int64_t> &shape,
    std::vector<T> &out)
{
    if(raw == nullptr)
    {
        throw std::invalid_argument(
            "copy_safetensors_to_nntile_layout: null raw buffer");
    }
    if(shape.empty())
    {
        throw std::invalid_argument(
            "copy_safetensors_to_nntile_layout: empty shape");
    }
    const Index vol = shape_volume(shape);
    const auto expected_bytes =
        static_cast<std::size_t>(vol) * sizeof(T);
    out.resize(static_cast<std::size_t>(vol));
    std::memcpy(out.data(), raw, expected_bytes);
}

template <typename T>
inline void read_tensor_nntile_layout(
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
            std::string("read_tensor_nntile_layout: byte size mismatch for ")
            + name);
    }
    copy_safetensors_to_nntile_layout<T>(raw.data(), info.shape, out);
}

} // namespace nntile::test::safetensors_nntile_layout
