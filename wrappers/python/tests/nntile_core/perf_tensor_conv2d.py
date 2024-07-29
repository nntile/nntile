# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/perf_tensor_conv2d.py
# Test for tensor::conv2d<T> Python wrapper
#
# @version 1.0.0

import itertools
import timeit

import click
import numpy as np

# All necesary imports
import nntile

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {
    np.float32: nntile.tensor.Tensor_fp32,
    np.float64: nntile.tensor.Tensor_fp64
}
# Define mapping between tested function and numpy type
conv2d = {
    np.float32: nntile.nntile_core.tensor.conv2d_fp32,
    np.float64: nntile.nntile_core.tensor.conv2d_fp64,
}


# Helper function returns bool value true if test passes
def helper(
    dtype,
    shape_A,
    shape_B,
    tile_shape_A,
    tile_shape_B,
    tile_shape_C,
    in_channels,
    out_channels,
    batch,
    padding,
):
    next_tag = 0

    bits = np.random.MT19937()
    rng = np.random.Generator(bits)
    shape = [*shape_A, in_channels, batch]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_A)
    mpi_distr = [0] * traits.grid.nelems
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    src_A = np.array(
            rng.standard_normal(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    next_tag = A.next_tag

    shape = [*shape_B, out_channels, in_channels]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_B)
    mpi_distr = [0] * traits.grid.nelems
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    src_B = np.array(
            rng(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    next_tag = B.next_tag

    shape = [
        shape_A[0] + shape_B[0] - 1 - 2 * padding[0],
        shape_A[1] + shape_B[1] - 1 - 2 * padding[1],
        out_channels,
        batch,
    ]
    traits = nntile.tensor.TensorTraits(shape, tile_shape_C)
    mpi_distr = [0] * traits.grid.nelems
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    src_C = np.array(
            rng(shape, dtype=dtype),
            dtype=dtype,
            order="F"
    )
    dst_C = np.zeros_like(src_C, dtype=dtype, order="F")

    # Set initial values of tensors
    A.from_array(src_A)
    B.from_array(src_B)
    C.from_array(src_C)

    iterations, timing = timeit.Timer(
        stmt="""
conv2d[dtype](A, B, C, padding[0], padding[1])
nntile.starpu.wait_for_all()""",
        globals=globals() | locals(),
    ).autorange()

    C.to_array(dst_C)
    A.unregister()
    B.unregister()
    C.unregister()

    return iterations, timing


@click.command()
@click.option(
    "--file", "-f", "file", type=click.File("a"),
    help="CSV file used for output"
)
@click.option(
    "--dtype",
    "-d",
    "dtype",
    multiple=True,
    type=click.Choice(["float32", "float64"]),
    help="type used for storage for all arrays",
)
@click.option(
    "--shape_A",
    "-A",
    "shape_A",
    multiple=True,
    type=click.Tuple([int, int]),
    help="shape of the src array",
)
@click.option(
    "--shape_B",
    "-B",
    "shape_B",
    multiple=True,
    type=click.Tuple([int, int]),
    help="shape of the kernel array",
)
@click.option(
    "--tile_shape_A",
    "-a",
    "tile_shape_A",
    multiple=True,
    type=click.Tuple([int, int, int, int]),
    help="shape of the single tile of src array",
)
@click.option(
    "--tile_shape_B",
    "-b",
    "tile_shape_B",
    multiple=True,
    type=click.Tuple([int, int, int, int]),
    help="shape of the single tile of kernel array",
)
@click.option(
    "--tile_shape_C",
    "-c",
    "tile_shape_C",
    multiple=True,
    type=click.Tuple([int, int, int, int]),
    help="shape of the single tile of dst array",
)
@click.option(
    "--in_channels",
    "-i",
    "in_channels",
    multiple=True,
    type=int,
    help="number of channels in the src array",
)
@click.option(
    "--out_channels",
    "-o",
    "out_channels",
    multiple=True,
    type=int,
    help="number of channels produced by the convolution",
)
@click.option(
    "--batch",
    "-r",
    "batch",
    multiple=True,
    type=int,
    help="butch size for all of the arrays",
)
@click.option(
    "--padding",
    "-p",
    "padding",
    multiple=True,
    type=click.Tuple([int, int]),
    help="padding added to 2 axes of the src (from both sides)",
)
def all_configurations(
    file,
    dtype,
    shape_A,
    shape_B,
    tile_shape_A,
    tile_shape_B,
    tile_shape_C,
    in_channels,
    out_channels,
    batch,
    padding,
):
    all_args = (
        dtype,
        shape_A,
        shape_B,
        tile_shape_A,
        tile_shape_B,
        tile_shape_C,
        in_channels,
        out_channels,
        batch,
        padding,
    )
    if file.tell() == 0:
        file.write(
            "time, dtype, Ax, Ay, Bx, By, tile_Ax, tile_Ay, tile_Aic, tile_Ab,"
            "tile_Bx, tile_By, tile_Bic, tile_Boc, tile_Cx, tile_Cy, tile_Coc,"
            "tile_Cb, in_channels, out_channels, batch, padding_x, padding_y\n"
        )
    for chosen_args in itertools.product(*all_args):
        str_arg, *num_args = chosen_args
        nptype = {"float32": np.float32, "float64": np.float64}[str_arg]
        iterations, timing = helper(nptype, *num_args)
        num_args = list(
            itertools.chain.from_iterable(
                [a if isinstance(a, tuple) else (a,) for a in num_args]
            )
        )
        file.write(f"{timing / iterations}, {str_arg}, "
                "{', '.join(map(str, num_args))}\n")


if __name__ == "__main__":
    all_configurations()
