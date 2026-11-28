<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file logger/README.md
 @version 2.0.0
-->

# nntile_logger_server

## Usage

The logger is an optional diagnostics service: NNTile's `log_scalar` codelet
connects to it over WebSocket and forwards one scalar at a time.

```bash
python3 -m pip install -r requirements.txt
SERVER_PORT=5001 LOG_DIR=logs SPLIT_HOURS=24 python3 server.py
# TensorBoard reads the event files under LOG_DIR
```

Enable logging in the NNTile context (`logger=1`, `logger_addr`, `logger_port`)
and keep the server running for the duration of the run. If the server is not
running, `log_scalar` is not submitted.
