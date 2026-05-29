/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/gpt2/gpt2_common.hh
 * Shared helpers for GPT-2 graph modules.
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>

namespace nntile::model::gpt2
{

//! Placeholder until built-in causal masking is implemented.
inline void throw_if_causal_flag_set(bool causal, const char* module_name)
{
    if(causal)
    {
        throw std::runtime_error(std::string(module_name) +
            ": causal=true is not yet implemented; pass an explicit BOOL "
            "attention mask for causal attention, or mask=nullptr for full "
            "bidirectional attention");
    }
}

} // namespace nntile::model::gpt2
