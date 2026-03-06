/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/io/safetensors.hh
 * SafeTensors reader/writer for tensor serialization.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nntile/graph/dtype.hh>

namespace nntile::io
{

//! Convert NNTile DataType to SafeTensors dtype string.
//! FP32_FAST_* variants all map to "F32" (storage type, not compute hint).
std::string dtype_to_safetensors(graph::DataType dtype);

//! Convert SafeTensors dtype string to NNTile DataType.
//! Returns the base storage type (e.g. "F32" -> FP32, not any FAST variant).
graph::DataType safetensors_to_dtype(const std::string& st_dtype);

//! Check whether a SafeTensors dtype string is a valid storage type
//! that NNTile can handle.
bool is_safetensors_dtype_compatible(const std::string& st_dtype,
                                     graph::DataType nntile_dtype);

//! Metadata for a single tensor stored in a SafeTensors file.
struct TensorInfo
{
    std::string name;
    graph::DataType dtype;
    std::vector<std::int64_t> shape;
    std::size_t data_offset;
    std::size_t data_size;
};

//! Reader for the SafeTensors binary format.
//!
//! On construction, reads and parses the JSON header from the file.
//! Individual tensors are read on demand via read_tensor(), which seeks
//! to the correct byte range -- enabling single-tensor access without
//! loading the whole file.
class SafeTensorsReader
{
public:
    //! Open a SafeTensors file and parse its header.
    //! @throws std::runtime_error if the file cannot be opened or the header
    //!         is malformed.
    explicit SafeTensorsReader(const std::string& path);

    //! List all tensor names in the file (order matches JSON key order).
    std::vector<std::string> tensor_names() const;

    //! Get metadata for a tensor by name.
    //! @throws std::runtime_error if the name is not found.
    const TensorInfo& tensor_info(const std::string& name) const;

    //! Check whether a tensor with the given name exists.
    bool has_tensor(const std::string& name) const;

    //! Read a single tensor's raw bytes from disk.
    //! Seeks to the tensor's byte range and reads exactly data_size bytes.
    //! @throws std::runtime_error if the name is not found or read fails.
    std::vector<std::uint8_t> read_tensor(const std::string& name) const;

    //! Number of tensors in the file.
    std::size_t size() const { return tensors_.size(); }

    //! Path of the underlying file.
    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::size_t header_size_ = 0;
    std::map<std::string, TensorInfo> tensors_;
};

//! Writer for the SafeTensors binary format.
//!
//! Accumulates tensors via add_tensor(), then writes the complete file
//! (JSON header + raw data) in one shot via write(). Can be used for a
//! single tensor or for a full module checkpoint.
class SafeTensorsWriter
{
public:
    SafeTensorsWriter() = default;

    //! Add a tensor to be written.
    //! @param name   Tensor name (must be unique within this writer).
    //! @param dtype  NNTile DataType (FP32_FAST_* variants stored as F32).
    //! @param shape  Tensor shape (NNTile column-major convention).
    //! @param data   Raw bytes in NNTile layout; size must equal
    //!               product(shape) * dtype_size(dtype).
    //! @throws std::invalid_argument if name is duplicate or data size mismatches.
    void add_tensor(const std::string& name,
                    graph::DataType dtype,
                    const std::vector<std::int64_t>& shape,
                    const std::vector<std::uint8_t>& data);

    //! Move-overload for zero-copy when caller can give up ownership.
    void add_tensor(const std::string& name,
                    graph::DataType dtype,
                    const std::vector<std::int64_t>& shape,
                    std::vector<std::uint8_t>&& data);

    //! Write all accumulated tensors to a SafeTensors file.
    //! @throws std::runtime_error if the file cannot be opened for writing.
    void write(const std::string& path) const;

    //! Number of tensors accumulated so far.
    std::size_t size() const { return entries_.size(); }

    //! Clear all accumulated tensors.
    void clear() { entries_.clear(); }

private:
    struct Entry
    {
        std::string name;
        graph::DataType dtype;
        std::vector<std::int64_t> shape;
        std::vector<std::uint8_t> data;
    };

    std::vector<Entry> entries_;
};

//! Save a single tensor to a SafeTensors file (convenience wrapper).
void save_tensor(const std::string& path,
                 const std::string& name,
                 graph::DataType dtype,
                 const std::vector<std::int64_t>& shape,
                 const std::vector<std::uint8_t>& data);

//! Load a single tensor's raw bytes from a SafeTensors file by name.
std::vector<std::uint8_t> load_tensor(const std::string& path,
                                       const std::string& name);

} // namespace nntile::io
