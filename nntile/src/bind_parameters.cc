#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/bind_parameters.cc
 *
 * @version 1.1.0
 * */

#include "nntile/bind_parameters.hh"

#include "nntile/dtype.hh"
#include "nntile/nn/graph_data_node.hh"
#include "nntile/runtime.hh"
#include "nntile/tile/graph.hh"

#include <stdexcept>

namespace nntile
{

void bind_tensor_host_data(Runtime &rt, NNGraph::TensorNode *tensor)
{
    if (tensor == nullptr)
    {
        throw std::invalid_argument(
            "bind_tensor_host_data: tensor must be non-null");
    }
    TensorGraph::TensorNode *data = tensor->data();
    const std::vector<std::uint8_t> *host = data->get_bind_hint();
    if (host == nullptr || host->empty())
    {
        throw std::runtime_error(
            "bind_tensor_host_data: no host data on tensor '" +
            tensor->name() + "'");
    }
    switch (data->dtype())
    {
    case DataType::FP32:
        rt.bind_data<float>(
            data,
            reinterpret_cast<const float *>(host->data()),
            host->size() / sizeof(float));
        break;
    case DataType::FP64:
        rt.bind_data<double>(
            data,
            reinterpret_cast<const double *>(host->data()),
            host->size() / sizeof(double));
        break;
    case DataType::FP16:
        rt.bind_data<float>(
            data,
            reinterpret_cast<const float *>(host->data()),
            host->size() / sizeof(nntile::fp16_t));
        break;
    case DataType::BF16:
        rt.bind_data<float>(
            data,
            reinterpret_cast<const float *>(host->data()),
            host->size() / sizeof(nntile::bf16_t));
        break;
    case DataType::INT64:
        rt.bind_data<std::int64_t>(
            data,
            reinterpret_cast<const std::int64_t *>(host->data()),
            host->size() / sizeof(std::int64_t));
        break;
    case DataType::BOOL:
        rt.bind_data<bool>(
            data,
            reinterpret_cast<const bool *>(host->data()),
            host->size() / sizeof(bool));
        break;
    default:
        throw std::runtime_error(
            "bind_tensor_host_data: unsupported dtype for '" +
            tensor->name() + "'");
    }
}

} // namespace nntile
