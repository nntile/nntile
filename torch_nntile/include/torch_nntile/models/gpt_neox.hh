/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt_neox.hh
 */

#pragma once

#include <torch_nntile/models/llama.hh>

namespace torch_nntile::models
{

//! GPT-NeoX tiny causal LM reuses the Llama LibTorch stack (RoPE) for v1.
using GptNeoXConfig = LlamaConfig;
using GptNeoXCausal = LlamaCausal;
using GptNeoXCausalImpl = LlamaCausalImpl;

} // namespace torch_nntile::models
