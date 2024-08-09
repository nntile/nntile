# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file logger/server.py
# A simple server that listens websocket port 5001 and writes TensorBoard logs
#
# @version 1.1.0

import asyncio
import datetime
import json
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

import tensorflow as tf

NODE_COUNTER = {}
WRITERS = {}
MEMORY_NODES_COUNTER_SEND = {}
MEMORY_NODES_COUNTER_RECEIVED = {}
MEMORY_NODES_COUNTER = {}
BYTES_TO_GB = 1 / (1024 * 1024 * 1024)


def create_new_writer(log_dir, node_name):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d---%H%M")
    current_log_dir = Path(log_dir) / node_name / current_time
    print(current_log_dir)
    Path(current_log_dir).mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(current_log_dir))
    writer.set_as_default()
    return writer


async def handle_new_logs(log_dir, split_hours):
    while True:
        await asyncio.sleep(split_hours * 60 * 60)
        for key in WRITERS.keys():
            WRITERS[key] = create_new_writer(log_dir, key)


def write_data(writer, tag, value, step):
    with writer.as_default():
        tf.summary.scalar(tag, value, step)


def increaseStep(node, node_dict):
    if node not in node_dict:
        node_dict[node] = 1
    else:
        node_dict[node] = node_dict[node] + 1


async def start_tensorboard(log_dir):
    print(Path.cwd())
    process = await asyncio.create_subprocess_exec(
            'tensorboard',
            '--logdir',
            log_dir,
            '--reload_multifile=true',
            '--bind_all',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print(f'TensorBoard stdout: {stdout.decode()}')
    print(f'TensorBoard stderr: {stderr.decode()}')


def handle_flops_message(workers_data, log_dir):
    for worker in workers_data:
        name = worker.get("name")
        time = float(worker.get("total_time"))
        flops = float(worker.get("flops"))
        if name not in WRITERS:
            WRITERS[name] = create_new_writer(log_dir, name)

        increaseStep(name, NODE_COUNTER)
        write_data(
                WRITERS[name],
                'Perf/GFlops/s',
                flops / time / 1e9,
                NODE_COUNTER[name]
        )


def handle_memory_nodes_message(memory_nodes_data, log_dir):
    for worker in memory_nodes_data:
        name = worker.get("name")
        size = float(worker.get("size"))
        if name not in WRITERS:
            WRITERS[name] = create_new_writer(log_dir, name)

        increaseStep(name, MEMORY_NODES_COUNTER)
        write_data(
                WRITERS[name],
                "MemoryUsage by GB",
                size * BYTES_TO_GB,
                MEMORY_NODES_COUNTER[name]
        )


def handle_bus_message(buses_data, log_dir):
    memory_nodes_sum_sent = {}
    memory_nodes_sum_received = {}
    for bus in buses_data:
        total_bus_time = float(bus.get("total_bus_time"))
        transferred_bytes = int(bus.get("transferred_bytes"))
        src = bus.get("src_name")
        dst = bus.get("dst_name")
        bus_speed_gbps = transferred_bytes / total_bus_time / 1e9
        bus_name = f"{src}->{dst}"
        if bus_name not in WRITERS:
            WRITERS[bus_name] = create_new_writer(log_dir, bus_name)

        increaseStep(bus_name, NODE_COUNTER)
        write_data(
                WRITERS[bus_name],
                'Perf/Data_link_GB/s',
                bus_speed_gbps,
                NODE_COUNTER[bus_name]
        )

        memory_nodes_sum_sent[src] = memory_nodes_sum_sent.get(src, 0) \
                + bus_speed_gbps
        memory_nodes_sum_received[dst] = \
                memory_nodes_sum_received.get(dst, 0) + bus_speed_gbps

    for name, speed in memory_nodes_sum_sent.items():
        if name not in WRITERS:
            WRITERS[name] = create_new_writer(log_dir, name)

        increaseStep(name, MEMORY_NODES_COUNTER_SEND)
        write_data(
                WRITERS[name],
                'Perf/Data_send_GB/s',
                speed,
                MEMORY_NODES_COUNTER_SEND[name]
        )

    for name, speed in memory_nodes_sum_received.items():
        if name not in WRITERS:
            WRITERS[name] = create_new_writer(log_dir, name)

        increaseStep(name, MEMORY_NODES_COUNTER_RECEIVED)
        write_data(
                WRITERS[name],
                'Perf/Data_recv_GB/s',
                speed,
                MEMORY_NODES_COUNTER_RECEIVED[name]
        )


async def handle_client(log_dir, reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"Connect from {addr}")
    while True:
        data = await reader.readline()
        if not data:
            break
        message = data.decode().strip()
        try:
            parsed_data = json.loads(message)
            workers_data = parsed_data.get("workers")
            buses_data = parsed_data.get("buses")
            memory_nodes_data = parsed_data.get("memory_nodes")
            handle_flops_message(workers_data, log_dir)
            handle_bus_message(buses_data, log_dir)
            handle_memory_nodes_message(memory_nodes_data, log_dir)
        except json.JSONDecodeError:
            print("Error decoding JSON:", message)


async def main():
    log_dir = os.environ.get('LOG_DIR', 'logs')
    split_hours = int(os.environ.get('SPLIT_HOURS', 24))
    clear_logs = int(os.environ.get('CLEAR_LOGS', 1))
    server_port = int(os.environ.get('SERVER_PORT', 5001))

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f"log_dir={log_dir}, split_hours={split_hours}")

    handle_client_log = partial(handle_client, log_dir)
    server = await asyncio.start_server(
        handle_client_log, "0.0.0.0", server_port)

    addr = server.sockets[0].getsockname()
    print(f"Server has been started on {addr}")

    async def start_server():
        async with server:
            await server.serve_forever()

    if clear_logs:
        shutil.rmtree(log_dir)

    await asyncio.gather(
        handle_new_logs(log_dir, split_hours),
        start_server(),
        start_tensorboard(log_dir)
    )


if __name__ == '__main__':
    asyncio.run(main())
