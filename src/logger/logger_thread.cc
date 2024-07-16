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
 * @version 1.0.0
 * */

#include "nntile/logger/logger_thread.hh"
#include <iostream>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <sys/socket.h>
#include <starpu.h>
#include "nntile/logger/websocket_client.hh"

namespace nntile::logger
{

//! Separate thread to log activities
std::thread logger_thread;
//! Flag if thread is running and is in good state
std::atomic<bool> logger_running(false);

//! Main routine for the logger thread
void logger_main()
{
    // At first get worker count, bus count and check if starpu is initialized
    int workerid;
    int busid;
    int worker_cnt = starpu_worker_get_count();
    int bus_cnt = starpu_bus_get_count();
    int is_initialized = starpu_is_initialized();
    std::cout << "WORKER COUNT: " << worker_cnt << std::endl;
    std::cout << "BUS COUNT: " << bus_cnt << std::endl;
    std::cout << "IS initialized : " << is_initialized << std::endl;
    // Infinite loop until NNTile exits this thread
    while (logger_running)
    {
        // Loop through all workers to get their activities
        for (workerid = 0; workerid < worker_cnt; workerid++)
        {
            // Profiling info is read from StarPU
            struct starpu_profiling_worker_info info;
            int ret = starpu_profiling_worker_get_info(workerid, &info);
            if (ret != 0)
                continue;
            // Get name of the worker
            // TODO: move this out of infinite loop
            char name[64];
            starpu_worker_get_name(workerid, name, sizeof(name));
            // Read how long the worker is running
            double total_time = starpu_timing_timespec_to_us(&info.total_time)
                * 1e-6;
            // Read how many FLOPs are performed by the worker
            double flops = 0.0;
            if (info.flops)
                flops = info.flops;
            // Form message to send
            char message[256];
            snprintf(message, sizeof(message), "{\"type\": \"0\", \"name\": \"%s\","
                    "\"total_time\": \"%.2lf\", \"flops\": \"%.2lf\"}\n",
                    name, total_time, flops);
            // Send the message
            if (send(client_socket, message, strlen(message), 0)
                    != (ssize_t)strlen(message))
            {
                perror("send");
            }
        }
        for (busid = 0; busid < bus_cnt; busid++)
        {
            int src, dst;
            char src_name[128], dst_name[128];
            // Profiling info is read from StarPU
            struct starpu_profiling_bus_info info;
            int ret = starpu_bus_get_profiling_info(busid, &info);
            if (ret != 0)
                continue;
            // Read the profiling information for the bus
            double total_bus_time = starpu_timing_timespec_to_us(&info.total_time)
                * 1e-6;
            uint64_t transferred_bytes = info.transferred_bytes;
            src = starpu_bus_get_src(busid);
		    dst = starpu_bus_get_dst(busid);
            starpu_memory_node_get_name(src, src_name, sizeof(src_name));
		    starpu_memory_node_get_name(dst, dst_name, sizeof(dst_name));        

            // Form message to send
            char message[512];
            snprintf(message, sizeof(message),
                "{ \"type\": \"1\", \"total_bus_time\": \"%.2lf\", "
                " \"transferred_bytes\": \"%llu\","
                " \"src_name\": \"%s\", \"dst_name\": \"%s\"}\n",
                total_bus_time, transferred_bytes, src_name, dst_name);
            // Send the message
            if (send(client_socket, message, strlen(message), 0) != (ssize_t)strlen(message))
            {
                perror("send");
            }
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

} // namespace nntile::logger
