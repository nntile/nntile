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

import os
import subprocess
import datetime
import shutil
import asyncio
import json
import tensorflow as tf

NODE_COUNTER = {}

async def create_new_writer(log_dir):
    global writer
    current_time = datetime.datetime.now().strftime("%Y-%m-%d---%H%M")
    current_log_dir = os.path.join(log_dir, current_time)
    print(current_log_dir)
    os.makedirs(current_log_dir, exist_ok=True)
    if "writer" in globals():
        writer.close()

    writer = tf.summary.create_file_writer(current_log_dir)
    writer.set_as_default()
    return writer

async def handle_new_logs(log_dir, split_hours):
    global writer
    global NODE_COUNTER
    NODE_COUNTER = {}
    if split_hours > 0:
        while True:
            writer = await create_new_writer(log_dir)
            # Convert hours into seconds
            await asyncio.sleep(split_hours * 60 * 60)
    else:
        writer = await create_new_writer(log_dir)


def increaseStep(node, node_dict):
    if not node in node_dict:
        node_dict[node] = 1
    else:
        node_dict[node] = node_dict[node] + 1

async def start_tensorboard(log_dir):
    print(os.getcwd())
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
            name = parsed_data.get("name")
            flops = float(parsed_data.get("flops"))
            increaseStep(name, NODE_COUNTER)
            tf.summary.scalar(name, flops, NODE_COUNTER[name])
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", message)

async def main():
    log_dir = os.environ['LOG_DIR']
    split_hours = int(os.environ['SPLIT_HOURS'])
    os.makedirs(log_dir, exist_ok=True)
    print(f"log_dir={log_dir}, split_hours={split_hours}")
    server = await asyncio.start_server(
        handle_client, "0.0.0.0", 5001)

    addr = server.sockets[0].getsockname()
    print(f"Server has been started on {addr}")

    async def start_server():
        async with server:
            await server.serve_forever()

    shutil.rmtree(log_dir)
    await asyncio.gather(
        handle_new_logs(log_dir, split_hours),
        start_server(),
        start_tensorboard(log_dir)
    )

if __name__ == '__main__':
    asyncio.run(main())
