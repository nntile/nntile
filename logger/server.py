# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file logger/server.py
# TCP server that listens on port 5001 and writes TensorBoard logs
#
# @version 1.1.0

import asyncio
import datetime
import json
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
from pathlib import Path

from tensorboardX import SummaryWriter

BYTES_TO_GB = 1 / (1024 * 1024 * 1024)


class LoggerState:
    """Holds all mutable state for the logger server."""

    def __init__(self):
        self.writers = {}
        self.node_counter = {}
        self.memory_nodes_counter_send = {}
        self.memory_nodes_counter_received = {}
        self.memory_nodes_counter = {}
        self.scalars_counter = {}

    def get_or_create_writer(self, log_dir: Path, node_name: str):
        """Get existing writer or create new one for the node."""
        if node_name not in self.writers:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d---%H%M")
            current_log_dir = log_dir / node_name / current_time
            current_log_dir.mkdir(parents=True, exist_ok=True)
            self.writers[node_name] = SummaryWriter(str(current_log_dir))
        return self.writers[node_name]

    def increase_step(self, node: str, counter_dict: dict) -> int:
        """Increment and return step for the given node."""
        counter_dict[node] = counter_dict.get(node, 0) + 1
        return counter_dict[node]

    def rotate_writers(self, log_dir: Path):
        """Rotate all writers: flush, close, and create new ones."""
        for key in list(self.writers.keys()):
            old_writer = self.writers[key]
            try:
                old_writer.flush()
                old_writer.close()
            except Exception as e:
                print(f"Warning: error closing writer for {key}: {e}")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d---%H%M")
            current_log_dir = log_dir / key / current_time
            current_log_dir.mkdir(parents=True, exist_ok=True)
            self.writers[key] = SummaryWriter(str(current_log_dir))

    def close_all(self):
        """Close all writers gracefully."""
        for key, writer in self.writers.items():
            try:
                writer.flush()
                writer.close()
            except Exception as e:
                print(f"Warning: error closing writer for {key}: {e}")
        self.writers.clear()


class LoggerServer:
    """Main server with proper lifecycle and shutdown handling."""

    def __init__(
        self,
        log_dir: str = "logs",
        split_hours: int = 24,
        clear_logs: bool = True,
        server_port: int = 5001,
        tensorboard_port: int = 6006,
        tls_cert_file: str | None = None,
        tls_key_file: str | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.split_hours = split_hours
        self.clear_logs = clear_logs
        self.server_port = server_port
        self.tensorboard_port = tensorboard_port
        self.tls_cert_file = tls_cert_file
        self.tls_key_file = tls_key_file
        self.state = LoggerState()
        self._server = None
        self._tensorboard_process = None
        self._tls_server = None
        self._shutdown_event = asyncio.Event()

    def _write_data(self, writer: SummaryWriter, tag: str, value: float, step: int):
        """Write scalar to TensorBoard."""
        writer.add_scalar(tag, value, step)

    def _handle_flops_message(self, workers_data: list, log_dir: Path):
        workers_data = workers_data or []
        for worker in workers_data:
            name = worker.get("name")
            total_time = float(worker.get("total_time", 1e-9))
            flops = float(worker.get("flops", 0))
            writer = self.state.get_or_create_writer(log_dir, name)
            step = self.state.increase_step(name, self.state.node_counter)
            self._write_data(writer, "Perf/GFlops/s", flops / total_time / 1e9, step)

    def _handle_memory_nodes_message(self, memory_nodes_data: list, log_dir: Path):
        memory_nodes_data = memory_nodes_data or []
        for worker in memory_nodes_data:
            name = worker.get("name")
            size = float(worker.get("size", 0))
            writer = self.state.get_or_create_writer(log_dir, name)
            step = self.state.increase_step(name, self.state.memory_nodes_counter)
            self._write_data(
                writer, "MemoryUsage by GB", size * BYTES_TO_GB, step
            )

    def _handle_bus_message(self, buses_data: list, log_dir: Path):
        buses_data = buses_data or []
        memory_nodes_sum_sent = {}
        memory_nodes_sum_received = {}
        for bus in buses_data:
            total_bus_time = float(bus.get("total_bus_time", 1e-9))
            transferred_bytes = int(bus.get("transferred_bytes", 0))
            src = bus.get("src_name", "")
            dst = bus.get("dst_name", "")
            bus_speed_gbps = transferred_bytes / total_bus_time / 1e9
            bus_name = f"{src}->{dst}"
            writer = self.state.get_or_create_writer(log_dir, bus_name)
            step = self.state.increase_step(bus_name, self.state.node_counter)
            self._write_data(
                writer, "Perf/Data_link_GB/s", bus_speed_gbps, step
            )
            memory_nodes_sum_sent[src] = (
                memory_nodes_sum_sent.get(src, 0) + bus_speed_gbps
            )
            memory_nodes_sum_received[dst] = (
                memory_nodes_sum_received.get(dst, 0) + bus_speed_gbps
            )
        for name, speed in memory_nodes_sum_sent.items():
            writer = self.state.get_or_create_writer(log_dir, name)
            step = self.state.increase_step(
                name, self.state.memory_nodes_counter_send
            )
            self._write_data(writer, "Perf/Data_send_GB/s", speed, step)
        for name, speed in memory_nodes_sum_received.items():
            writer = self.state.get_or_create_writer(log_dir, name)
            step = self.state.increase_step(
                name, self.state.memory_nodes_counter_received
            )
            self._write_data(writer, "Perf/Data_recv_GB/s", speed, step)

    def _handle_scalars(self, scalars_data: list, log_dir: Path):
        """Handle user scalars. Uses scalar name as tag for TensorBoard hierarchy.
        Example: 'Train loss' -> tag 'Train loss', visible under Scalars in TB.
        For hierarchical organization use slashes: 'Training/loss', 'Validation/acc'
        """
        scalars_data = scalars_data or []
        for scalar in scalars_data:
            name = scalar.get("name", "scalar")
            values = scalar.get("values") or []
            writer = self.state.get_or_create_writer(log_dir, name)
            for value in values:
                step = self.state.increase_step(name, self.state.scalars_counter)
                self._write_data(writer, name, float(value), step)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"Connect from {addr}")
        try:
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
                    scalars_data = parsed_data.get("scalars")
                    if scalars_data:
                        self._handle_scalars(scalars_data, self.log_dir)
                    self._handle_flops_message(workers_data, self.log_dir)
                    self._handle_bus_message(buses_data, self.log_dir)
                    self._handle_memory_nodes_message(
                        memory_nodes_data, self.log_dir
                    )
                except json.JSONDecodeError:
                    print("Error decoding JSON:", message)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_log_rotation(self):
        """Periodically rotate log writers."""
        last_rotation = asyncio.get_event_loop().time()
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1)
            if self._shutdown_event.is_set():
                break
            now = asyncio.get_event_loop().time()
            if (now - last_rotation) >= self.split_hours * 3600:
                self.state.rotate_writers(self.log_dir)
                last_rotation = now

    async def _shutdown_waiter(self):
        """Wait for shutdown signal and close servers."""
        await self._shutdown_event.wait()
        if self._server:
            self._server.close()
        if self._tls_server:
            self._tls_server.close()

    async def _wait_for_tensorboard(self, timeout: float = 30.0) -> bool:
        """Wait until TensorBoard is listening on its port."""
        start = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start) < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(("127.0.0.1", self.tensorboard_port))
                sock.close()
                if result == 0:
                    print("TensorBoard is ready")
                    return True
            except (socket.error, OSError):
                pass
            await asyncio.sleep(0.5)
        print("Warning: TensorBoard did not become ready in time", file=sys.stderr)
        return False

    async def _start_tensorboard(self):
        """Start TensorBoard subprocess."""
        cmd = [
            "tensorboard",
            "--logdir",
            str(self.log_dir),
            "--port",
            str(self.tensorboard_port),
            "--bind_all",
        ]
        self._tensorboard_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await self._wait_for_tensorboard()
        if self._tensorboard_process.returncode is not None:
            stdout, stderr = await self._tensorboard_process.communicate()
            print(f"TensorBoard exited: {stdout.decode()}")
            print(f"TensorBoard stderr: {stderr.decode()}")

    async def _run_tls_proxy(self):
        """Run TLS proxy for TensorBoard if cert/key are configured."""
        if not self.tls_cert_file or not self.tls_key_file:
            return
        tls_port = int(os.environ.get("TLS_PORT", "6443"))
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(
            self.tls_cert_file,
            self.tls_key_file,
        )

        async def tls_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ):
            try:
                backend_reader, backend_writer = await asyncio.open_connection(
                    "127.0.0.1", self.tensorboard_port
                )
                async def forward(src, dst):
                    try:
                        while True:
                            data = await src.read(8192)
                            if not data:
                                break
                            dst.write(data)
                            await dst.drain()
                    except (ConnectionResetError, BrokenPipeError):
                        pass
                    finally:
                        dst.close()

                await asyncio.gather(
                    forward(reader, backend_writer),
                    forward(backend_reader, writer),
                )
            except Exception as e:
                print(f"TLS proxy error: {e}")
            finally:
                writer.close()
                await writer.wait_closed()

        self._tls_server = await asyncio.start_server(
            tls_handler, "0.0.0.0", tls_port, ssl=ssl_context
        )
        print(f"TLS proxy listening on https://0.0.0.0:{tls_port}")

    async def run(self):
        """Run the logger server with proper lifecycle."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"log_dir={self.log_dir}, split_hours={self.split_hours}")

        if self.clear_logs and self.log_dir.exists():
            shutil.rmtree(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        def shutdown_handler(*_):
            print("Shutdown signal received")
            self._shutdown_event.set()

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, shutdown_handler)
            except NotImplementedError:
                pass

        self._server = await asyncio.start_server(
            self._handle_client, "0.0.0.0", self.server_port
        )
        addr = self._server.sockets[0].getsockname()
        print(f"Logger server listening on {addr}")

        await self._start_tensorboard()
        if self.tls_cert_file and self.tls_key_file:
            await self._run_tls_proxy()

        async def serve():
            async with self._server:
                await self._server.serve_forever()

        tasks = [
            self._shutdown_waiter(),
            serve(),
            self._handle_log_rotation(),
        ]
        if self._tls_server:
            tasks.append(self._tls_server.serve_forever())

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown()
            self._shutdown_event.set()

    def _shutdown(self):
        """Clean shutdown of all resources."""
        print("Shutting down logger server...")
        if self._server:
            self._server.close()
        self.state.close_all()
        if self._tensorboard_process and self._tensorboard_process.returncode is None:
            self._tensorboard_process.terminate()
            try:
                self._tensorboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._tensorboard_process.kill()
        if self._tls_server:
            self._tls_server.close()
        print("Logger server stopped")


async def main():
    log_dir = os.environ.get("LOG_DIR", "logs")
    split_hours = int(os.environ.get("SPLIT_HOURS", 24))
    clear_logs = bool(int(os.environ.get("CLEAR_LOGS", 1)))
    server_port = int(os.environ.get("SERVER_PORT", 5001))
    tensorboard_port = int(os.environ.get("TENSORBOARD_PORT", 6006))
    tls_cert = os.environ.get("TLS_CERT_FILE")
    tls_key = os.environ.get("TLS_KEY_FILE")

    server = LoggerServer(
        log_dir=log_dir,
        split_hours=split_hours,
        clear_logs=clear_logs,
        server_port=server_port,
        tensorboard_port=tensorboard_port,
        tls_cert_file=tls_cert,
        tls_key_file=tls_key,
    )
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
