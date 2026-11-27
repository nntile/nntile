/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/tests/kernel/chunk_class_probe.cc
 * Assert the live chunk class matches the configured environment. Without
 * this, a suite run at NNHAUL_CHUNK_SIZE_MB=1 would stay green even if the
 * knob stopped reaching the chunk class (the adaptive tests would just see
 * single-chunk tiles and still pass).
 *
 * SPDX-License-Identifier: MIT
 *
 * @version 1.1.0
 * */

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main()
{
    char const *mb = std::getenv("NNHAUL_CHUNK_SIZE_MB");
    if(mb == nullptr || *mb == '\0')
    {
        std::printf("chunk_class_probe: no class override; nothing to assert\n");
        return 0;
    }
    std::size_t const expected =
        std::size_t(std::strtoul(mb, nullptr, 10)) << 20;
    std::size_t const got = nntile::kernel::chunk_class().mask + 1;
    if(got != expected)
    {
        std::fprintf(stderr,
            "chunk_class_probe: class is %zu bytes, environment asks for "
            "%zu\n", got, expected);
        return 1;
    }
    std::printf("chunk_class_probe: %zu-byte class matches the environment\n",
        got);
    return 0;
}
