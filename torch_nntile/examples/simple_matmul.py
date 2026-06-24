"""This is an example of parallel distributed matrix multiplications using
torch_nntile extension of the torch"""

# Imports at first, torch_nntile enables device="nntile" backend of torch
import time
import torch
import torch_nntile


# Provide matrix multiplciation shapes
M = 10240
N = 9216
K = 10240
REPEAT = 100

# Define tiles via int or List[int]
# We do not split tensors into tiles yet due to some bug.
MT = M
NT = N
KT = K
# Once the bug is resolved, we can effectively split into tiles
#MT = 2048
#NT = 3072
#KT = [1024, 2048, 3072, 4096] # These numbers must sum up to K

# Lazy initalization of parameters
#   ncpu=0 means we use no CPU cores for computations
#   ncuda=-1 means we use all CUDA gpus, available via CUDA_VISIBLE_DEVICES
#   verbose=1 is to show us when NNTile is initialized or finalized
#   runtime_mode="graph" is to capture the computational graph to execute it
#       manually. Setting this to "eager" executes every operation nearly
#       immediately. However, "eager" mode does not support tiling.
torch_nntile.init_context(ncpu=0, ncuda=-1, verbose=1, runtime_mode="graph")
torch_nntile.restrict_cuda()

# Define inputs by Pytorch
a = torch.randn(M, K)
b = torch.randn(K, N)

# Copy inputs to NNTile backend
a_nnt = a.to(device="nntile")
b_nnt = b.to(device="nntile")

# Compile an empty graph to prefetch data to the NNTile
t = -time.time()
torch_nntile.compile_graph()
t += time.time()
print(f"Prefetch data time: {t} s")
torch_nntile.run()
torch_nntile.wait()

# Ask to multiply inputs (no computations happen here yet)
for i in range(REPEAT):
    c_nnt = a_nnt @ b_nnt

# Provide convenient names to axis groups:
#   a_nnt is M by K
#   b_nnt is K by N
#   c_nnt is M by N
# It is enough to provide axis names only to a_nnt and b_nnt,
# as corresponding axes are captured by torch_nntile automatically
torch_nntile.set_axis_group_name(a_nnt, {0: "M", 1: "K"})
torch_nntile.set_axis_group_name(b_nnt, {1: "N"})

# Define tiling schemes along each axis group
torch_nntile.set_axis_group_tiling("M", MT)
torch_nntile.set_axis_group_tiling("N", NT)
torch_nntile.set_axis_group_tiling("K", KT)

# Print some info on axis groups
torch_nntile.print_axis_groups()

# Compile the graph. It is nearly immedate, as it does no optimizations yet
t = -time.time()
torch_nntile.compile_graph()
t += time.time()
print(f"Graph compile time: {t} s")

# Now, finally, launch the computations
t = -time.time()
torch_nntile.run()
torch_nntile.wait()
t += time.time()

# Get the reference result
c = a.cuda() @ b.cuda()
c2 = c_nnt.cpu().cuda()

# Print the information
print(f"Matmul {REPEAT} times {M}x{K} @ {K}x{N} -> {M}x{N}: {t} s")
print(f"Performance: {2e-12 * REPEAT * M * K * N / t} Tflops/s")
print(f"Relative error vs Pytorch: {torch.norm(c-c2)/torch.norm(c)}")
