# NNTile Logger Server

Standalone TCP server that receives profiling and scalar data from NNTile applications and writes them to TensorBoard-compatible log files. Users can then visualize the data via TensorBoard over HTTP or HTTPS.

## Architecture

- **C++ client** (in NNTile): Background thread collects StarPU worker/bus/memory metrics and user scalars, sends JSON over TCP
- **Python server** (this): Listens on TCP port 5001, writes TensorBoard event files
- **TensorBoard**: Serves the logs via web UI on port 6006 (HTTP) or 6443 (HTTPS with TLS)

## Quick Start

### Run locally

```bash
cd logger
pip install -r requirements.txt
python server.py
```

### Run with Docker

```bash
docker build -t nntile_logger_server .
docker run -p 5001:5001 -p 6006:6006 nntile_logger_server
```

### Connect from NNTile

```python
import nntile
ctx = nntile.Context(ncpu=4, ncuda=1, logger=1, logger_addr="localhost", logger_port=5001)
# ... training ...
ctx.shutdown()
```

Then open http://localhost:6006 in your browser for TensorBoard.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_DIR` | `logs` | Directory for TensorBoard event files |
| `SERVER_PORT` | `5001` | TCP port for receiving log data from NNTile |
| `TENSORBOARD_PORT` | `6006` | Port for TensorBoard HTTP UI |
| `SPLIT_HOURS` | `24` | Rotate log files every N hours |
| `CLEAR_LOGS` | `1` | Clear existing logs on startup (1=yes, 0=no) |
| `TLS_CERT_FILE` | - | Path to TLS certificate (enables HTTPS proxy) |
| `TLS_KEY_FILE` | - | Path to TLS private key |
| `TLS_PORT` | `6443` | Port for HTTPS when TLS is enabled |

## TLS / HTTPS

For secure access to TensorBoard, set `TLS_CERT_FILE` and `TLS_KEY_FILE`. The server will start an HTTPS proxy on port 6443 that forwards to TensorBoard.

### Generate self-signed certificate (development)

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

Then run:

```bash
TLS_CERT_FILE=cert.pem TLS_KEY_FILE=key.pem python server.py
```

Access TensorBoard at https://localhost:6443

### Production

Use certificates from your CA or Let's Encrypt. Mount the cert and key into the container and set the env vars.

## TensorBoard Scalar Organization

User scalars (e.g. from `log_scalar_async("Train loss", value)`) are organized as follows:

- **Tag**: The scalar name is used as the TensorBoard tag
- **Hierarchy**: Use slashes for grouping, e.g. `"Training/loss"`, `"Validation/accuracy"`
- **Runs**: Each worker/bus/memory node gets its own run; user scalars are grouped by name

Example tags in TensorBoard:
- `Perf/GFlops/s` – worker performance
- `Perf/Data_link_GB/s` – bus throughput
- `MemoryUsage by GB` – memory usage
- `Train loss` – user scalar (use any name you pass to `log_scalar_async`)

## Docker Compose

When using `compose.yaml`, the logger service exposes:
- **5001**: Logger TCP (for NNTile to connect)
- **6006**: TensorBoard HTTP
- **6443**: TensorBoard HTTPS (when TLS certs are configured)

Other services (e.g. Jupyter) can connect to the logger at `logger-server:5001` on the Docker network.
