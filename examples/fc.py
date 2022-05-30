#!/usr/bin/env python

import wrappers.python.starpu as st
import wrappers.python.tensor as nt
import wrappers.python.layer as lt
import numpy as np
import time

starpu = st.Starpu()

n_layers = 10
n_batch = 4096
n_seq = 4096
n_batch_tile = 512
n_seq_tile = 512

# States in numpy
X = [np.random.randn(n_seq, n_batch).astype(np.float32)]
X.extend([np.ndarray((n_seq, n_batch), dtype=np.float32) for i in
    range(n_layers+1)])
# States in nntile
X_nntile = [nt.Tensor_fp32([n_seq, n_batch], [n_seq_tile, n_batch_tile]) for i in
        range(n_layers+1)]
X_nntile[0].from_array(X[0])
# Linear layers in numpy
FC = [np.random.randn(n_seq, n_seq).astype(np.float32, order='F') / n_seq**0.5 for i in
        range(n_layers)]
FC_nntile = [lt.FC_fp32([n_seq, n_seq], [n_seq_tile, n_seq_tile]) for i in
        range(n_layers)]
for i in range(n_layers):
    FC_nntile[i].from_array(FC[i])
# Start the timer
starpu.wait_for_all()
print("Start timer")
time0 = time.perf_counter()
# Launch the code
for i in range(n_layers):
    FC_nntile[i].forward_async(X_nntile[i], X_nntile[i+1])
# Finish the timer
starpu.wait_for_all()
print("Finish timer")
time1 = time.perf_counter()
dtime = time1 - time0
# Output results
gflops = n_layers * 2 * n_batch * n_seq**2 / 10**9
print("Gflops    : {}".format(gflops))
print("Time      : {}".format(dtime))
print("Gflops/s  : {}".format(gflops/dtime))
# Compare result
time0 = time.perf_counter()
for i in range(n_layers):
    X[i+1] = FC[i] @ X[i]
time1 = time.perf_counter()
dtime = time1 - time0
print("Numpy time: P={}".format(dtime))
Y = np.ndarray((n_seq, n_batch), dtype=np.float32, order='F')
X_nntile[n_layers].to_array(Y)
Z = X[n_layers]
print("diff/norm : {}".format(np.linalg.norm(Y-Z) / np.linalg.norm(Z)))
# Unregister all the NNTile data
for i in range(n_layers):
    X_nntile[i].unregister()
    FC_nntile[i].unregister()
X_nntile[n_layers].unregister()

