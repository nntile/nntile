/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/logger/websocket_client.hh
 * Simple web socket client
 *
 * @version 1.1.0
 * */

#pragma once

#include <stddef.h>

namespace nntile::logger
{

extern int client_socket;

void websocket_connect(const char *server_addr, int server_port);

void websocket_disconnect();

} // namespace nntile::logger
