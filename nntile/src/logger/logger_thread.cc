/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/logger/logger_thread.cc
 * Implementation of a logger thread
 *
 * @version 1.1.0
 * */

#include "nntile/logger/logger_thread.hh"
#include "nntile/defs.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <atomic>
#include <map>
#include <vector>
#include <mutex>
#include <unistd.h>
#include <sys/socket.h>
#include "nntile/logger/websocket_client.hh"

namespace nntile::logger
{

//! Separate thread to log activities
std::thread logger_thread;
//! Flag if thread is running and is in good state
std::atomic<bool> logger_running(false);
//! Mutex for work with scalar map
std::mutex scalars_mutex;
//! Scalar map
std::map<std::string, std::vector<float>> scalars;

//! Main routine for the logger thread
void logger_main()
{
    // At first get worker count, bus count and check if NNHaul is initialized
    int workerid;
    int busid;
    int worker_cnt = 0;
    int bus_cnt = 0;
    int is_initialized = 1;
    unsigned memnodes_cnt = 0;
    std::cout << "WORKER COUNT: " << worker_cnt << std::endl;
    std::cout << "BUS COUNT: " << bus_cnt << std::endl;
    std::cout << "MEMNODES COUNT: " << memnodes_cnt << std::endl;
    std::cout << "IS initialized : " << is_initialized << std::endl;
    // Infinite loop until NNTile exits this thread
    while (logger_running)
    {
        std::stringstream ss;
        ss << "{";
        ss << "\"workers\":[";
        bool first_worker = true;
        // Loop through all workers to get their activities
        for (workerid = 0; workerid < worker_cnt; workerid++)
        {
        }
        ss << "],";

        // Loop through buses (virtual links from one device to another)
        ss << "\"buses\":[";
        bool first_bus = true;
        for (busid = 0; busid < bus_cnt; busid++)
        {
        }
        ss << "],";

        // Get memory usage information for each memory node
        ss << "\"memory_nodes\":[";
        bool first_memory_node = true;
        for (unsigned memory_node = 0; memory_node < memnodes_cnt; memory_node++)
        {
        }
        ss << "]";

        // Read logged scalar values
        std::unique_lock<std::mutex> lock(scalars_mutex);
        if (!scalars.empty())
        {
            ss << ",\"scalars\":[";
            bool first_scalar = true;
            for (auto &[name, values] : scalars)
            {
                if (!values.empty())
                {
                    if (!first_scalar)
                        ss << ",";
                    first_scalar = false;

                    ss << "{";
                    ss << "\"name\":\"" << name << "\",";
                    ss << "\"values\":[";

                    bool first_value = true;
                    for (float value : values)
                    {
                        if (!first_value)
                            ss << ",";
                        first_value = false;
                        ss << value;
                    }

                    ss << "]}";

                    values.clear();
                }
            }
            ss << "]";
        }
        ss << "}";
        //Clear scalar map
        scalars.clear();
        //Unlock mutex
        lock.unlock();
        // Serialize JSON object to string
        std::string message = ss.str() + "\n";
        // Send the message
        if (send(client_socket, message.c_str(), message.length(), 0) != (ssize_t)message.length())
        {
            perror("send");
        }
        // Wait for 0.5 seconds until next time we read activities
        usleep(500000);
    }
}

//! Initialization of the logger thread
void logger_init(const char *server_addr, int server_port)
{
    // Connect to websocket
    websocket_connect(server_addr, server_port);
    // Start main logger thread function
    logger_running = true;
    logger_thread = std::thread(logger_main);
}

//! Finalize logger thread
void logger_shutdown()
{
    std::cout << "LOGGER SHUTDOWN" << std::endl;
    logger_running = false;
    if (logger_thread.joinable())
    {
        logger_thread.join();
    }
    websocket_disconnect();
}

//! Log scalar to scalar map
void log_scalar(const std::string &name, float value)
{
    std::lock_guard<std::mutex> lock(scalars_mutex);
    if (scalars.find(name) == scalars.end())
    {
        scalars[name] = std::vector<float>();
    }
    scalars[name].push_back(value);
}

} // namespace nntile::logger
