/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/logger/tcp_client.cc
 * TCP client for logger connection
 *
 * @version 1.1.0
 * */

#include "nntile/logger/tcp_client.hh"
#include <cerrno>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

namespace nntile::logger
{

int client_socket = -1;

bool tcp_connect(const char *server_addr, int server_port)
{
    struct sockaddr_in server;

    std::cout << "Logger: connecting to " << server_addr << ":" << server_port
              << std::endl;

    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1)
    {
        std::cerr << "Logger: socket creation failed: " << strerror(errno)
                  << std::endl;
        return false;
    }

    server.sin_family = AF_INET;
    server.sin_port = htons(static_cast<uint16_t>(server_port));
    if (inet_pton(AF_INET, server_addr, &server.sin_addr) <= 0)
    {
        std::cerr << "Logger: invalid server address" << std::endl;
        close(client_socket);
        client_socket = -1;
        return false;
    }

    if (connect(client_socket, reinterpret_cast<struct sockaddr *>(&server),
                sizeof(server)) < 0)
    {
        std::cerr << "Logger: connection failed: " << strerror(errno)
                  << " (logger disabled, application continues)" << std::endl;
        close(client_socket);
        client_socket = -1;
        return false;
    }

    std::cout << "Logger: connected successfully" << std::endl;
    return true;
}

void tcp_disconnect()
{
    if (client_socket != -1)
    {
        close(client_socket);
        client_socket = -1;
    }
}

bool tcp_is_connected()
{
    return client_socket != -1;
}

} // namespace nntile::logger
