/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/logger/logger_thread.cc
 * Implementation of a logger thread
 *
 * @version 1.1.0
 * */

#include "nntile/logger/logger_thread.hh"
#include "nntile/logger/tcp_client.hh"
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <map>
#include <vector>
#include <mutex>
#include <chrono>
#include <unistd.h>
#include <sys/socket.h>
#include <starpu.h>

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

//! Reconnection parameters
constexpr int RECONNECT_BASE_DELAY_MS = 1000;
constexpr int RECONNECT_MAX_DELAY_MS = 60000;
constexpr int RECONNECT_MAX_ATTEMPTS = 10;

//! Attempt reconnection with exponential backoff
bool try_reconnect(const char *server_addr, int server_port)
{
    for (int attempt = 0; attempt < RECONNECT_MAX_ATTEMPTS && logger_running;
         ++attempt)
    {
        int delay_ms = std::min(
            RECONNECT_BASE_DELAY_MS * (1 << attempt),
            RECONNECT_MAX_DELAY_MS);
        std::cerr << "Logger: reconnecting in " << delay_ms << " ms (attempt "
                  << (attempt + 1) << "/" << RECONNECT_MAX_ATTEMPTS << ")"
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

        if (tcp_connect(server_addr, server_port))
        {
            return true;
        }
    }
    return false;
}

//! Main routine for the logger thread
void logger_main(const char *server_addr, int server_port)
{
    // At first get worker count, bus count and check if starpu is initialized
    int worker_cnt = starpu_worker_get_count();
    int bus_cnt = starpu_bus_get_count();
    int is_initialized = starpu_is_initialized();
    unsigned memnodes_cnt = starpu_memory_nodes_get_count();
    std::cout << "WORKER COUNT: " << worker_cnt << std::endl;
    std::cout << "BUS COUNT: " << bus_cnt << std::endl;
    std::cout << "MEMNODES COUNT: " << memnodes_cnt << std::endl;
    std::cout << "IS initialized : " << is_initialized << std::endl;

    while (logger_running)
    {
        if (!tcp_is_connected())
        {
            if (!try_reconnect(server_addr, server_port))
            {
                std::cerr << "Logger: max reconnection attempts reached, "
                             "stopping logger thread"
                          << std::endl;
                break;
            }
        }

        nlohmann::json j;
        j["workers"] = nlohmann::json::array();
        j["buses"] = nlohmann::json::array();
        j["memory_nodes"] = nlohmann::json::array();

        // Loop through all workers to get their activities
        for (int workerid = 0; workerid < worker_cnt; workerid++)
        {
            struct starpu_profiling_worker_info info;
            int ret = starpu_profiling_worker_get_info(workerid, &info);
            if (ret != 0)
                continue;

            char name[64];
            starpu_worker_get_name(workerid, name, sizeof(name));
            double total_time =
                starpu_timing_timespec_to_us(&info.total_time) * 1e-6;
            double flops = info.flops ? info.flops : 0.0;

            j["workers"].push_back({{"name", name},
                                   {"total_time", total_time},
                                   {"flops", flops}});
        }

        // Loop through buses (virtual links from one device to another)
        for (int busid = 0; busid < bus_cnt; busid++)
        {
            int src, dst;
            char src_name[128], dst_name[128];
            struct starpu_profiling_bus_info info;
            int ret = starpu_bus_get_profiling_info(busid, &info);
            if (ret != 0)
                continue;

            double total_bus_time =
                starpu_timing_timespec_to_us(&info.total_time) * 1e-6;
            uint64_t transferred_bytes = info.transferred_bytes;
            src = starpu_bus_get_src(busid);
            dst = starpu_bus_get_dst(busid);
            starpu_memory_node_get_name(src, src_name, sizeof(src_name));
            starpu_memory_node_get_name(dst, dst_name, sizeof(dst_name));

            j["buses"].push_back(
                {{"total_bus_time", total_bus_time},
                 {"transferred_bytes", transferred_bytes},
                 {"src_name", src_name},
                 {"dst_name", dst_name}});
        }

        // Get memory usage information for each memory node
        for (unsigned memory_node = 0; memory_node < memnodes_cnt;
             memory_node++)
        {
            char memory_node_name[128];
            starpu_memory_node_get_name(memory_node, memory_node_name,
                                        sizeof(memory_node_name));
            size_t node = starpu_memory_get_used(memory_node);
            j["memory_nodes"].push_back(
                {{"name", memory_node_name}, {"size", node}});
        }

        // Read logged scalar values
        {
            std::lock_guard<std::mutex> lock(scalars_mutex);
            if (!scalars.empty())
            {
                j["scalars"] = nlohmann::json::array();
                for (auto &[name, values] : scalars)
                {
                    if (!values.empty())
                    {
                        j["scalars"].push_back(
                            {{"name", name}, {"values", values}});
                        values.clear();
                    }
                }
                scalars.clear();
            }
        }

        std::string message = j.dump() + "\n";

        ssize_t sent =
            send(client_socket, message.c_str(), message.length(), 0);
        if (sent != static_cast<ssize_t>(message.length()))
        {
            if (sent < 0)
            {
                perror("Logger send");
            }
            tcp_disconnect();
        }

        usleep(500000);
    }
}

//! Initialization of the logger thread
void logger_init(const char *server_addr, int server_port)
{
    if (!tcp_connect(server_addr, server_port))
    {
        std::cout << "Logger: server unavailable, logging disabled"
                  << std::endl;
        return;
    }

    logger_running = true;
    logger_thread = std::thread(logger_main, server_addr, server_port);
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
    tcp_disconnect();
}

//! Log scalar to scalar map
void log_scalar(const std::string &name, float value)
{
    std::lock_guard<std::mutex> lock(scalars_mutex);
    scalars[name].push_back(value);
}

} // namespace nntile::logger
