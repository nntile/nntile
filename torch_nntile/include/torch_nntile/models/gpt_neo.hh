/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt_neo.hh
 */

#pragma once

#include <torch_nntile/models/gpt2.hh>

namespace torch_nntile::models
{

//! GPT-Neo tiny causal LM reuses the GPT-2 LibTorch stack for v1 parity.
using GptNeoConfig = Gpt2Config;
using GptNeoCausal = Gpt2Causal;
using GptNeoCausalImpl = Gpt2CausalImpl;

} // namespace torch_nntile::models
