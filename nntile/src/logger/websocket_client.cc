/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/logger/websocket_client.cc
 * Simple web socket client
 *
 * @version 1.1.0
 * */

#include "nntile/logger/websocket_client.hh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <iostream>

namespace nntile::logger
{

int client_socket = -1;

void websocket_connect(const char *server_addr, int server_port)
{
    struct sockaddr_in server;

    std::cout << "Trying to connect to " << server_addr << ":" << server_port
        << "\n";
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1)
    {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    server.sin_family = AF_INET;
    server.sin_port = htons(server_port);
    inet_pton(AF_INET, server_addr, &server.sin_addr);

    if (connect(client_socket, (struct sockaddr *)&server, sizeof(server)) < 0)
    {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }
}

void websocket_disconnect()
{
    if (client_socket != -1)
    {
        close(client_socket);
        client_socket = -1;
    }
}

} // namespace nntile::namespace
