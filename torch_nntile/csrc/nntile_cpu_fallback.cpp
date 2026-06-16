/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_cpu_fallback.cpp
 * Self-contained CPU fallback for unsupported PrivateUse1 ops.
 *
 * macOS PyTorch wheels do not export at::native::cpu_fallback from
 * libtorch_cpu.dylib, so we cannot link against that symbol.
 */

#include <ATen/core/alias_info.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#include <c10/core/Device.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <optional>
#include <sstream>
#include <vector>

namespace torch_nntile
{

namespace
{

c10::Device nntile_device()
{
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

template <
    typename T,
    std::enable_if_t<
        std::is_same_v<T, at::Tensor> ||
            std::is_same_v<T, std::optional<at::Tensor>>,
        int> = 0>
std::vector<T> to_cpu(const std::vector<T> &tensors)
{
    const int num = static_cast<int>(tensors.size());
    std::vector<T> cpu_tensors(static_cast<size_t>(num));
    std::vector<at::Tensor> valid_tensors;
    std::vector<bool> to_translate(static_cast<size_t>(num), false);
    for (const auto i : c10::irange(num))
    {
        if constexpr (std::is_same_v<T, std::optional<at::Tensor>>)
        {
            if (tensors[static_cast<size_t>(i)].has_value() &&
                tensors[static_cast<size_t>(i)].value().defined())
            {
                to_translate[static_cast<size_t>(i)] = true;
                valid_tensors.push_back(tensors[static_cast<size_t>(i)].value());
            }
            else
            {
                cpu_tensors[static_cast<size_t>(i)] =
                    tensors[static_cast<size_t>(i)];
            }
        }
        else
        {
            if (tensors[static_cast<size_t>(i)].defined())
            {
                to_translate[static_cast<size_t>(i)] = true;
                valid_tensors.push_back(tensors[static_cast<size_t>(i)]);
            }
            else
            {
                cpu_tensors[static_cast<size_t>(i)] =
                    tensors[static_cast<size_t>(i)];
            }
        }
    }
    const auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
    for (int i = 0, defined_pos = 0; i < num; ++i)
    {
        if (to_translate[static_cast<size_t>(i)])
        {
            cpu_tensors[static_cast<size_t>(i)] =
                std::move(cpu_valid_tensors[static_cast<size_t>(defined_pos++)]);
        }
    }
    return cpu_tensors;
}

std::optional<c10::Device> compute_target_device(
    std::vector<at::Tensor> &t_args,
    const std::vector<c10::List<at::Tensor>> &tlist_args)
{
    if (!t_args.empty())
    {
        return t_args[0].device();
    }
    for (const auto &tens_list : tlist_args)
    {
        for (const auto i : c10::irange(tens_list.size()))
        {
            return tens_list.get(i).device();
        }
    }
    return std::nullopt;
}

bool validate_tensor_list(const c10::List<at::Tensor> &tensorlist)
{
    for (const auto i : c10::irange(tensorlist.size()))
    {
        if (tensorlist.get(i).defined())
        {
            return true;
        }
    }
    return false;
}

at::Tensor to_nntile(const at::Tensor &tensor, const c10::Device &device)
{
    return tensor.to(device);
}

} // namespace

void cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack)
{
    constexpr bool error_on_views = false;
    constexpr c10::DispatchKey cpu_dispatch_key = c10::DispatchKey::CPU;

    auto &schema_args = op.schema().arguments();
    const auto num_arguments = schema_args.size();
    auto arguments = torch::jit::last(*stack, num_arguments);
    const auto arguments_begin = stack->size() - num_arguments;

    std::vector<at::Tensor> tensor_args;
    std::vector<int64_t> tensor_args_indices;

    std::vector<c10::List<at::Tensor>> tensorlist_args;
    std::vector<int64_t> tensorlist_args_indices;

    std::vector<c10::List<std::optional<at::Tensor>>> optional_tensorlist_args;
    std::vector<int64_t> optional_tensorlist_args_indices;

    std::optional<c10::Device> tgt_device = std::nullopt;
    std::vector<c10::IValue> tensorlist_cpu_args;
    std::vector<c10::IValue> optional_tensorlist_cpu_args;

    for (const auto idx : c10::irange(arguments.size()))
    {
        const auto &ivalue = arguments[idx];
        if (ivalue.isTensor())
        {
            tensor_args.push_back(ivalue.toTensor());
            tensor_args_indices.push_back(static_cast<int64_t>(idx));
        }
        else if (ivalue.isTensorList())
        {
            tensorlist_args.push_back(ivalue.toTensorList());
            tensorlist_args_indices.push_back(static_cast<int64_t>(idx));
            auto cpu_ivalue = c10::IValue(
                c10::List<at::Tensor>(to_cpu(ivalue.toTensorVector())));
            tensorlist_cpu_args.push_back(cpu_ivalue);
            (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
        }
        else if (ivalue.isOptionalTensorList())
        {
            optional_tensorlist_args.push_back(ivalue.toOptionalTensorList());
            optional_tensorlist_args_indices.push_back(
                static_cast<int64_t>(idx));
            auto cpu_ivalue = c10::IValue(c10::List<std::optional<at::Tensor>>(
                to_cpu(ivalue.toOptionalTensorVector())));
            optional_tensorlist_cpu_args.push_back(cpu_ivalue);
            (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
        }
        else if (ivalue.isDevice())
        {
            tgt_device = ivalue.toDevice();
            (*stack)[arguments_begin + idx] = c10::IValue(c10::Device(c10::kCPU));
        }
    }

    const auto cpu_tensors = to_cpu(tensor_args);
    for (const auto i : c10::irange(tensor_args_indices.size()))
    {
        const auto idx = tensor_args_indices[i];
        (*stack)[arguments_begin + static_cast<size_t>(idx)] =
            c10::IValue(cpu_tensors[i]);
    }

    op.redispatchBoxed(c10::DispatchKeySet(cpu_dispatch_key), stack);

    for (const auto i : c10::irange(tensor_args_indices.size()))
    {
        const auto tensor_idx = tensor_args_indices[i];
        const c10::AliasInfo *alias_info =
            schema_args[static_cast<size_t>(tensor_idx)].alias_info();
        if (alias_info != nullptr && alias_info->isWrite())
        {
            if (!tensor_args[i].defined())
            {
                continue;
            }
            at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
        }
    }

    for (const auto i : c10::irange(tensorlist_args_indices.size()))
    {
        const auto tensorlist_idx = tensorlist_args_indices[i];
        const c10::AliasInfo *alias_info =
            schema_args[static_cast<size_t>(tensorlist_idx)].alias_info();
        if (alias_info != nullptr && alias_info->isWrite())
        {
            const auto &cpu_tens = tensorlist_cpu_args[i].toTensorVector();
            for (const auto idx : c10::irange(tensorlist_args[i].size()))
            {
                if (!cpu_tens[idx].defined())
                {
                    continue;
                }
                at::_copy_from_and_resize(
                    cpu_tens[idx], tensorlist_args[i][idx]);
            }
        }
    }

    for (const auto i : c10::irange(optional_tensorlist_args_indices.size()))
    {
        const auto tensorlist_idx = optional_tensorlist_args_indices[i];
        const c10::AliasInfo *alias_info =
            schema_args[static_cast<size_t>(tensorlist_idx)].alias_info();
        if (alias_info != nullptr && alias_info->isWrite())
        {
            const auto &cpu_tens =
                optional_tensorlist_cpu_args[i].toOptionalTensorList();
            for (const auto idx :
                c10::irange(optional_tensorlist_args[i].size()))
            {
                if (cpu_tens[idx].has_value())
                {
                    const std::optional<at::Tensor> dst =
                        optional_tensorlist_args[i][idx];
                    if (dst.has_value() && dst->defined())
                    {
                        at::_copy_from_and_resize(
                            *cpu_tens[idx], *dst);
                    }
                }
            }
        }
    }

    const auto &schema_returns = op.schema().returns();
    const auto num_returns = schema_returns.size();
    auto returns = torch::jit::last(*stack, num_returns);
    const auto returns_begin = stack->size() - num_returns;

    if (!tgt_device.has_value())
    {
        tgt_device = compute_target_device(tensor_args, tensorlist_args);
    }
    if (!tgt_device.has_value())
    {
        tgt_device = nntile_device();
    }

    for (const auto idx : c10::irange(returns.size()))
    {
        const c10::AliasInfo *alias_info =
            schema_returns[static_cast<size_t>(idx)].alias_info();
        if (alias_info != nullptr && alias_info->isWrite())
        {
            bool found_alias = false;
            if (returns[idx].isTensor() && returns[idx].toTensor().defined())
            {
                for (const auto i : c10::irange(tensor_args_indices.size()))
                {
                    const auto input_tensor_idx = tensor_args_indices[i];
                    const auto &input_tensor = cpu_tensors[i];
                    const c10::AliasInfo *input_alias_info =
                        schema_args[static_cast<size_t>(input_tensor_idx)]
                            .alias_info();
                    if (input_tensor.defined() &&
                        (alias_info == input_alias_info ||
                            (input_alias_info != nullptr &&
                                *alias_info == *input_alias_info)))
                    {
                        (*stack)[returns_begin + idx] =
                            c10::IValue(tensor_args[i]);
                        found_alias = true;
                        break;
                    }
                }
            }
            else if (
                returns[idx].isTensorList() &&
                validate_tensor_list(returns[idx].toTensorList()))
            {
                for (const auto i : c10::irange(tensorlist_args_indices.size()))
                {
                    const auto input_tensor_idx = tensorlist_args_indices[i];
                    const c10::AliasInfo *input_alias_info =
                        schema_args[static_cast<size_t>(input_tensor_idx)]
                            .alias_info();
                    if (validate_tensor_list(tensorlist_args[i]) &&
                        (alias_info == input_alias_info ||
                            (input_alias_info != nullptr &&
                                *alias_info == *input_alias_info)))
                    {
                        (*stack)[returns_begin + idx] =
                            c10::IValue(tensorlist_args[i]);
                        found_alias = true;
                        break;
                    }
                }
            }
            TORCH_CHECK(
                found_alias,
                "The operator ",
                op.schema().operator_name(),
                " appears to have invalid alias information.");
        }
        else
        {
            if (alias_info != nullptr && !alias_info->isWrite())
            {
                std::stringstream dev_str;
                dev_str << *tgt_device;
                if (error_on_views)
                {
                    TORCH_CHECK(
                        false,
                        "The operator ",
                        op.schema().operator_name(),
                        " appears to be a view operator, but it has no "
                        "implementation for the backend \"",
                        dev_str.str(),
                        "\".");
                }
                else
                {
                    TORCH_WARN(
                        "The operator ",
                        op.schema().operator_name(),
                        " appears to be a view operator, but it has no "
                        "implementation for the backend \"",
                        dev_str.str(),
                        "\". View operators don't support falling back to "
                        "run on the CPU.");
                }
            }
            if (returns[idx].isTensor() && returns[idx].toTensor().defined())
            {
                (*stack)[returns_begin + idx] = c10::IValue(
                    to_nntile(returns[idx].toTensor(), *tgt_device));
            }
            else if (
                returns[idx].isTensorList() &&
                validate_tensor_list(returns[idx].toTensorList()))
            {
                const auto &cpu_tensors_out =
                    returns[idx].toTensorList().vec();
                std::vector<at::Tensor> tensors;
                tensors.reserve(cpu_tensors_out.size());
                for (const auto &tensor : cpu_tensors_out)
                {
                    tensors.push_back(to_nntile(tensor, *tgt_device));
                }
                (*stack)[returns_begin + idx] =
                    c10::IValue(c10::List<at::Tensor>(tensors));
            }
        }
    }
}

} // namespace torch_nntile
