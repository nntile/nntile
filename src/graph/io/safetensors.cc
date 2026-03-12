/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/io/safetensors.cc
 * SafeTensors reader/writer implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/io/safetensors.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nntile::graph::io
{

// -------------------------------------------------------------------------
// Dtype mapping
// -------------------------------------------------------------------------

std::string dtype_to_safetensors(DataType dtype)
{
    switch(dtype)
    {
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
            return "F32";
        case DataType::FP64:
            return "F64";
        case DataType::FP16:
            return "F16";
        case DataType::BF16:
            return "BF16";
        case DataType::INT64:
            return "I64";
        case DataType::BOOL:
            return "BOOL";
        default:
            throw std::invalid_argument(
                "dtype_to_safetensors: unknown DataType");
    }
}

DataType safetensors_to_dtype(const std::string& st_dtype)
{
    if(st_dtype == "F32")  return DataType::FP32;
    if(st_dtype == "F64")  return DataType::FP64;
    if(st_dtype == "F16")  return DataType::FP16;
    if(st_dtype == "BF16") return DataType::BF16;
    if(st_dtype == "I64")  return DataType::INT64;
    if(st_dtype == "BOOL") return DataType::BOOL;
    throw std::invalid_argument(
        "safetensors_to_dtype: unknown SafeTensors dtype '" + st_dtype + "'");
}

bool is_safetensors_dtype_compatible(const std::string& st_dtype,
                                     DataType nntile_dtype)
{
    return dtype_to_safetensors(nntile_dtype) == st_dtype;
}

// -------------------------------------------------------------------------
// SafeTensorsReader
// -------------------------------------------------------------------------

SafeTensorsReader::SafeTensorsReader(const std::string& path)
    : path_(path)
{
    std::ifstream file(path_, std::ios::binary);
    if(!file)
    {
        throw std::runtime_error(
            "SafeTensorsReader: cannot open file '" + path_ + "'");
    }

    // Read 8-byte little-endian header size
    std::uint64_t raw_header_size = 0;
    file.read(reinterpret_cast<char*>(&raw_header_size), 8);
    if(!file || file.gcount() != 8)
    {
        throw std::runtime_error(
            "SafeTensorsReader: failed to read header size from '" +
            path_ + "'");
    }
    header_size_ = static_cast<std::size_t>(raw_header_size);

    if(header_size_ == 0)
    {
        return;
    }

    // Read JSON header
    std::string header_json(header_size_, '\0');
    file.read(header_json.data(),
              static_cast<std::streamsize>(header_size_));
    if(!file)
    {
        throw std::runtime_error(
            "SafeTensorsReader: failed to read header JSON from '" +
            path_ + "'");
    }

    // Parse JSON header
    nlohmann::json header;
    try
    {
        header = nlohmann::json::parse(header_json);
    }
    catch(const nlohmann::json::parse_error& e)
    {
        throw std::runtime_error(
            "SafeTensorsReader: JSON parse error in '" + path_ +
            "': " + e.what());
    }

    // Extract tensor metadata
    for(auto it = header.begin(); it != header.end(); ++it)
    {
        const std::string& name = it.key();

        // Skip the optional "__metadata__" key
        if(name == "__metadata__")
        {
            continue;
        }

        const auto& entry = it.value();

        TensorInfo info;
        info.name = name;

        // dtype (accept string "F32" or numeric code for robustness)
        if(!entry.contains("dtype"))
        {
            throw std::runtime_error(
                "SafeTensorsReader: tensor '" + name +
                "' missing 'dtype' in '" + path_ + "'");
        }
        std::string st_dtype;
        const auto& dtype_val = entry["dtype"];
        if(dtype_val.is_string())
        {
            st_dtype = dtype_val.get<std::string>();
        }
        else if(dtype_val.is_number_integer())
        {
            int code = dtype_val.get<int>();
            switch(code)
            {
                case 0: st_dtype = "F32"; break;
                case 1: st_dtype = "F16"; break;
                case 2: st_dtype = "F64"; break;
                case 3: st_dtype = "BF16"; break;
                case 4: st_dtype = "I64"; break;
                default:
                    throw std::runtime_error(
                        "SafeTensorsReader: tensor '" + name +
                        "' has unknown dtype code " + std::to_string(code) +
                        " in '" + path_ + "' (expected string like \"F32\")");
            }
        }
        else
        {
            throw std::runtime_error(
                "SafeTensorsReader: tensor '" + name +
                "' dtype must be string or int in '" + path_ + "'");
        }
        info.dtype = safetensors_to_dtype(st_dtype);

        // shape
        if(!entry.contains("shape"))
        {
            throw std::runtime_error(
                "SafeTensorsReader: tensor '" + name +
                "' missing 'shape' in '" + path_ + "'");
        }
        for(const auto& dim : entry["shape"])
        {
            info.shape.push_back(dim.get<std::int64_t>());
        }

        // data_offsets
        if(!entry.contains("data_offsets"))
        {
            throw std::runtime_error(
                "SafeTensorsReader: tensor '" + name +
                "' missing 'data_offsets' in '" + path_ + "'");
        }
        const auto& offsets = entry["data_offsets"];
        if(offsets.size() != 2)
        {
            throw std::runtime_error(
                "SafeTensorsReader: tensor '" + name +
                "' data_offsets must have exactly 2 elements in '" +
                path_ + "'");
        }
        info.data_offset = offsets[0].get<std::size_t>();
        std::size_t data_end = offsets[1].get<std::size_t>();
        info.data_size = data_end - info.data_offset;

        tensors_.emplace(name, std::move(info));
    }
}

std::vector<std::string> SafeTensorsReader::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for(const auto& [name, _] : tensors_)
    {
        names.push_back(name);
    }
    return names;
}

const TensorInfo& SafeTensorsReader::tensor_info(
    const std::string& name) const
{
    auto it = tensors_.find(name);
    if(it == tensors_.end())
    {
        throw std::runtime_error(
            "SafeTensorsReader::tensor_info: tensor '" + name +
            "' not found in '" + path_ + "'");
    }
    return it->second;
}

bool SafeTensorsReader::has_tensor(const std::string& name) const
{
    return tensors_.count(name) > 0;
}

std::vector<std::uint8_t> SafeTensorsReader::read_tensor(
    const std::string& name) const
{
    const auto& info = tensor_info(name);

    std::ifstream file(path_, std::ios::binary);
    if(!file)
    {
        throw std::runtime_error(
            "SafeTensorsReader::read_tensor: cannot open file '" +
            path_ + "'");
    }

    // Data begins at byte 8 + header_size_ + data_offset
    const std::size_t file_offset = 8 + header_size_ + info.data_offset;
    file.seekg(static_cast<std::streamoff>(file_offset));
    if(!file)
    {
        throw std::runtime_error(
            "SafeTensorsReader::read_tensor: seek failed for tensor '" +
            name + "' in '" + path_ + "'");
    }

    std::vector<std::uint8_t> data(info.data_size);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(info.data_size));
    if(!file)
    {
        throw std::runtime_error(
            "SafeTensorsReader::read_tensor: read failed for tensor '" +
            name + "' in '" + path_ + "'");
    }

    return data;
}

// -------------------------------------------------------------------------
// SafeTensorsWriter
// -------------------------------------------------------------------------

namespace
{

std::size_t compute_expected_size(
    const std::vector<std::int64_t>& shape,
    DataType dtype)
{
    std::int64_t nelems = 1;
    for(auto dim : shape)
    {
        nelems *= dim;
    }
    return static_cast<std::size_t>(nelems) * dtype_size(dtype);
}

} // anonymous namespace

void SafeTensorsWriter::add_tensor(
    const std::string& name,
    DataType dtype,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::uint8_t>& data)
{
    add_tensor(name, dtype, shape, std::vector<std::uint8_t>(data));
}

void SafeTensorsWriter::add_tensor(
    const std::string& name,
    DataType dtype,
    const std::vector<std::int64_t>& shape,
    std::vector<std::uint8_t>&& data)
{
    for(const auto& e : entries_)
    {
        if(e.name == name)
        {
            throw std::invalid_argument(
                "SafeTensorsWriter::add_tensor: duplicate name '" +
                name + "'");
        }
    }

    std::size_t expected = compute_expected_size(shape, dtype);
    if(data.size() != expected)
    {
        throw std::invalid_argument(
            "SafeTensorsWriter::add_tensor: data size mismatch for '" +
            name + "': expected " + std::to_string(expected) +
            " bytes, got " + std::to_string(data.size()));
    }

    entries_.push_back({name, dtype, shape, std::move(data)});
}

void SafeTensorsWriter::write(const std::string& path) const
{
    // Build JSON header: compute data offsets for each tensor
    nlohmann::json header = nlohmann::json::object();
    std::size_t running_offset = 0;

    for(const auto& entry : entries_)
    {
        nlohmann::json tensor_meta;
        tensor_meta["dtype"] = dtype_to_safetensors(entry.dtype);

        nlohmann::json shape_arr = nlohmann::json::array();
        for(auto dim : entry.shape)
        {
            shape_arr.push_back(dim);
        }
        tensor_meta["shape"] = shape_arr;

        std::size_t data_end = running_offset + entry.data.size();
        tensor_meta["data_offsets"] = {running_offset, data_end};
        running_offset = data_end;

        header[entry.name] = tensor_meta;
    }

    // Serialize header to compact JSON
    std::string header_json = header.dump();
    std::uint64_t header_size =
        static_cast<std::uint64_t>(header_json.size());

    // Write file
    std::ofstream file(path, std::ios::binary);
    if(!file)
    {
        throw std::runtime_error(
            "SafeTensorsWriter::write: cannot open file '" + path +
            "' for writing");
    }

    // 8-byte little-endian header size
    file.write(reinterpret_cast<const char*>(&header_size), 8);

    // JSON header
    file.write(header_json.data(),
               static_cast<std::streamsize>(header_json.size()));

    // Tensor data in order
    for(const auto& entry : entries_)
    {
        file.write(reinterpret_cast<const char*>(entry.data.data()),
                   static_cast<std::streamsize>(entry.data.size()));
    }

    if(!file)
    {
        throw std::runtime_error(
            "SafeTensorsWriter::write: write error for file '" + path + "'");
    }
}

// -------------------------------------------------------------------------
// Convenience free functions
// -------------------------------------------------------------------------

void save_tensor(const std::string& path,
                 const std::string& name,
                 DataType dtype,
                 const std::vector<std::int64_t>& shape,
                 const std::vector<std::uint8_t>& data)
{
    SafeTensorsWriter writer;
    writer.add_tensor(name, dtype, shape, data);
    writer.write(path);
}

std::vector<std::uint8_t> load_tensor(const std::string& path,
                                       const std::string& name)
{
    SafeTensorsReader reader(path);
    return reader.read_tensor(name);
}

} // namespace nntile::graph::io
