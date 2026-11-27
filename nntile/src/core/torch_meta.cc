/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/core/torch_meta.cc
 * Shared tile meta helpers for torch-native ops.
 *
 * @version 1.1.0
 */

#include "nntile/core/torch_meta.hh"

namespace nntile::core
{

TorchTileMeta make_contiguous_torch_meta(const std::vector<Index> &sizes)
{
    TorchTileMeta meta;
    meta.sizes = sizes;
    meta.strides.resize(sizes.size());
    if (sizes.empty())
    {
        return meta;
    }
    meta.strides.back() = 1;
    for (Index i = static_cast<Index>(sizes.size()) - 2; i >= 0; --i)
    {
        meta.strides[static_cast<size_t>(i)] =
            meta.strides[static_cast<size_t>(i + 1)] *
            sizes[static_cast<size_t>(i + 1)];
    }
    return meta;
}

} // namespace nntile::core
