/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/logger/tcp_client.hh
 * TCP client for logger connection
 *
 * @version 1.1.0
 * */

#pragma once

#include <stddef.h>

namespace nntile::logger
{

extern int client_socket;

//! Connect to the logger server
//! @return true if connection succeeded, false otherwise (logger remains optional)
bool tcp_connect(const char *server_addr, int server_port);

//! Disconnect from the logger server
void tcp_disconnect();

//! Check if connected
bool tcp_is_connected();

} // namespace nntile::logger
