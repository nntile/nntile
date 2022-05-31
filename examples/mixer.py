#!/usr/bin/env python

import wrappers.python.starpu as st
import wrappers.python.tensor as nt
import wrappers.python.layer as lt
import numpy as np
import time

starpu = st.Starpu()

n_iters = 1 # Number of repeats of entire scheme
n_layers = 1 # Number of mixer layers
n_b = 1024 # Number of images in batch
n_b_tile = 512 # Number of images per tile
n_p = 256 # Number of patches
n_p_tile = 256 # Number of patches per tile
n_c = 1280 # Number of channels of each patch
n_c_tile = 320 # Number of channels per tile
n_p_mlp = 640 # Size of MLP extension for patches
n_p_mlp_tile = 320 # Tile size of MLP extension for patches
n_c_mlp = 5120 # Size of MLP extension for channels
n_c_mlp_tile = 512 # Tile size of MLP extension for channels
eps = 1e-5 ** 0.5 # Regularization for the normalization

# Inputs/outputs of the mixer layers
X_nntile = [nt.Tensor_fp32([n_p, n_b, n_c], [n_p_tile, n_b_tile, n_c_tile])
        for i in range(n_layers+1)]
X = np.random.randn(n_p, n_b, n_c)
X_nntile[0].from_array(X)
del X
# Mixer layers
MX_nntile = [lt.Mixer_fp32(n_p, n_p_tile, n_p_mlp, n_p_mlp_tile, n_c, n_c_tile,
    n_c_mlp, n_c_mlp_tile, n_b, n_b_tile, eps) for i in range(n_layers)]
# Numpy arrays to init mixer layers
MX = [(np.random.randn(n_p_mlp, n_p) / n_p_mlp**0.5,
    np.random.randn(n_p, n_p_mlp) / n_p**0.5,
    np.random.randn(n_c, n_c_mlp) / n_c**0.5,
    np.random.randn(n_c_mlp, n_c) / n_c_mlp**0.5) for i in range(n_layers)]
for i in range(n_layers):
    MX_nntile[i].from_arrays(MX[i][0], MX[i][1], MX[i][2], MX[i][3])
del MX
# Start the timer
starpu.wait_for_all()
print("Start timer")
time0 = time.perf_counter()
# Run mixer
for it in range(n_iters):
    for i in range(n_layers):
        MX_nntile[i].forward_async(X_nntile[i], X_nntile[i+1])
# Finish the timer
starpu.wait_for_all()
print("Finish timer")
time1 = time.perf_counter()
dtime = (time1-time0) / n_iters
# Output results
print("Avg.time  : {}".format(dtime))
# Unregister all the NNTile data
starpu.wait_for_all()
for i in range(n_layers):
    X_nntile[i].unregister()
    MX_nntile[i].unregister()
X_nntile[n_layers].unregister()
del X_nntile
del MX_nntile
