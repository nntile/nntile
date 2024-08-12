/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/logger/logger_thread.hh
 * Headers for logger thread
 *
 * @version 1.1.0
 * */

#pragma once

#include <atomic>

namespace nntile::logger
{

extern std::atomic<bool> logger_running;

void logger_init(const char *server_addr, int server_port);

void logger_shutdown();

} // namespace nntile::logger
