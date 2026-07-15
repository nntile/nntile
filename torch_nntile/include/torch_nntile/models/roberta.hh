/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/roberta.hh
 */

#pragma once

#include <torch_nntile/models/bert.hh>

namespace torch_nntile::models
{

//! RoBERTa MLM reuses BERT LibTorch stack (no token-type in Python; C++ shares).
using RobertaConfig = BertConfig;
using RobertaMlm = BertMlm;
using RobertaMlmImpl = BertMlmImpl;

} // namespace torch_nntile::models
