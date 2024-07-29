#!/usr/bin/bash

export STARPU_NCPU=2 STARPU_SILENT=1 STARPU_PROFILING=1 STARPU_FXT_TRACE=0

export LOGGER_SERVER=$(getent hosts logger-server | awk '{ print $1 }')

python wrappers/python/examples/gpt2_training.py --nntile-nepochs=100 \
    --nntile-logger --nntile-logger-server-addr=${LOGGER_SERVER}
