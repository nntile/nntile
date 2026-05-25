/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/graph/model/bert/bert_common.hh
 * Shared helpers for BERT graph modules.
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>

namespace nntile::model::bert
{

inline void throw_if_causal_flag_set(bool causal, const char* module_name)
{
    if(causal)
    {
        throw std::runtime_error(std::string(module_name) +
            ": causal=true is not supported for BERT encoder attention");
    }
}

} // namespace nntile::model::bert
