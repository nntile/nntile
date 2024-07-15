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
# @version 1.0.0

import asyncio
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

import tensorflow as tf

NODE_COUNTER = {}
BUS_COUNTER = {}
WRITERS = {}
LOG_DIR = "logs"


async def create_new_writer(log_dir, node_name):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d---%H%M")
    current_log_dir = Path(log_dir) / node_name / current_time
    print(current_log_dir)
    Path(current_log_dir).mkdir(parents=True)

    writer = tf.summary.create_file_writer(current_log_dir)
    writer.set_as_default()
    return writer


async def handle_new_logs(log_dir, split_hours):
    global WRITERS
    global NODE_COUNTER
    global BUS_COUNTER
    NODE_COUNTER = {}
    BUS_COUNTER = {}
    WRITERS = {}
    while True:
        await asyncio.sleep(split_hours * 60 * 60)
        for key in WRITERS.keys():
            WRITERS[key] = await create_new_writer(log_dir, key)


def increaseStep(node, node_dict):
    if node not in node_dict:
        node_dict[node] = 1
    else:
        node_dict[node] = node_dict[node] + 1


async def start_tensorboard(log_dir):
    print(Path.getcwd())
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


async def handle_flops_message(parsed_data, log_dir):
    name = parsed_data.get("name")
    flops = float(parsed_data.get("flops"))

    if name not in WRITERS:
        WRITERS[name] = await create_new_writer(log_dir, name)

    with WRITERS[name].as_default():
        increaseStep(name, NODE_COUNTER)
        tf.summary.scalar("Flops", flops, NODE_COUNTER[name])


async def handle_bus_message(parsed_data, log_dir):
    bus_id = parsed_data.get("bus_id")
    total_bus_time = float(parsed_data.get("total_bus_time"))
    transferred_bytes = int(parsed_data.get("transferred_bytes"))
    transfer_count = int(parsed_data.get("transfer_count"))

    if bus_id not in WRITERS:
        WRITERS[bus_id] = await create_new_writer(log_dir, f"bus_{bus_id}")

    with WRITERS[bus_id].as_default():
        increaseStep(bus_id, BUS_COUNTER)
        tf.summary.scalar(f"Bus/{bus_id}/Total_Bus_Time",
                total_bus_time,
                BUS_COUNTER[bus_id]
        )
        tf.summary.scalar(
                f"Bus/{bus_id}/Transferred_Bytes",
                transferred_bytes,
                BUS_COUNTER[bus_id]
        )
        tf.summary.scalar(
                f"Bus/{bus_id}/Transfer_Count",
                transfer_count,
                BUS_COUNTER[bus_id]
        )


async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"Connect from {addr}")
    while True:
        data = await reader.readline()
        if not data:
            break
        message = data.decode().strip()
        try:
            parsed_data = json.loads(message)
            message_type = int(parsed_data.get("type"))
            match message_type:
                case 0:
                    await handle_flops_message(parsed_data, LOG_DIR)
                case 1:
                    await handle_bus_message(parsed_data, LOG_DIR)
                case _:
                    print(f"Unknown message type: {message_type}")
        except json.JSONDecodeError:
            print("Error decoding JSON:", message)


async def main():
    log_dir = os.environ.get('LOG_DIR', 'logs')
    global LOG_DIR
    LOG_DIR = log_dir
    split_hours = int(os.environ.get('SPLIT_HOURS', 24))
    clear_logs = int(os.environ.get('CLEAR_LOGS', 1))
    server_port = int(os.environ.get('SERVER_PORT', 5001))

    Path(log_dir).mkdir(parents=True)
    print(f"log_dir={log_dir}, split_hours={split_hours}")
    server = await asyncio.start_server(
        handle_client, "0.0.0.0", server_port)

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
