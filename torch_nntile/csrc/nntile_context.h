/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_context.h
 */

#pragma once

#include <torch_nntile/runtime.hh>

namespace torch_nntile
{

void ensure_nntile_context();

//! Apply the torch-session chunked-tile policy at session start: with
//! NNHAUL_CHUNK_TILES unset, disable chunked tile allocation for the
//! whole session (torch_dispatch ops cannot consume chunk chains; see
//! the session law in nntile/nnhaul/chunk_gate.hh). Must run BEFORE the
//! first tile allocates — called from context creation and from the
//! platform-session start.
void apply_torch_session_chunk_policy();

//! True when libnntile was built on the NNHaul runtime backend
//! (NNTILE_USE_NNHAUL) instead of StarPU.
bool uses_nnhaul();

} // namespace torch_nntile
