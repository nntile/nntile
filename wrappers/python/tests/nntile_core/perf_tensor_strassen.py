# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/perf_tensor_strassen.py
# Test for tensor::strassen<T> Python wrapper
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
Tensor = {np.float32: nntile.tensor.Tensor_fp32, np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
strassen = {
    np.float32: nntile.nntile_core.tensor.strassen_fp32,
    np.float64: nntile.nntile_core.tensor.strassen_fp64,
}


# Helper function returns bool value true if test passes
def helper(dtype, tA, tB, matrix_shape, shared_size, tile_size, alpha, beta, batches):
    # Describe single-tile tensor, located at node 0
    mpi_distr = [0]
    next_tag = 0
    tile_shape = [tile_size, tile_size] + [1] * len(batches)

    if tA == nntile.notrans:
        shape = [matrix_shape[0], shared_size, *batches]
    else:
        shape = [shared_size, matrix_shape[0], *batches]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    A = Tensor[dtype](traits, mpi_distr, next_tag)
    src_A = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = A.next_tag

    if tB == nntile.notrans:
        shape = [shared_size, matrix_shape[1], *batches]
    else:
        shape = [matrix_shape[1], shared_size, *batches]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    B = Tensor[dtype](traits, mpi_distr, next_tag)
    src_B = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    next_tag = B.next_tag

    shape = [matrix_shape[0], matrix_shape[1], *batches]
    traits = nntile.tensor.TensorTraits(shape, tile_shape)
    mpi_distr = [0] * traits.grid.nelems
    C = Tensor[dtype](traits, mpi_distr, next_tag)
    src_C = np.array(np.random.randn(*shape), dtype=dtype, order="F")
    dst_C = np.zeros_like(src_C, dtype=dtype, order="F")

    # Set initial values of tensors
    A.from_array(src_A)
    B.from_array(src_B)
    C.from_array(src_C)

    C.to_array(dst_C)
    iterations, timing = timeit.Timer(
        stmt="""
strassen[dtype](alpha, tA, A, tB, B, beta, C, 1, len(batches), 0)
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
    "--file", "-f", "file", type=click.File("a"), help="CSV file used for output"
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
    "--matrix_shape",
    "-m",
    "matrix_shape",
    multiple=True,
    type=click.Tuple([int, int]),
    help="shape of the resulting array",
)
@click.option(
    "--tA",
    "-a",
    "tA",
    multiple=True,
    type=click.Choice(["notrans", "trans"]),
    help="transpose of A array",
)
@click.option(
    "--tB",
    "-b",
    "tB",
    multiple=True,
    type=click.Choice(["notrans", "trans"]),
    help="transpose of B array",
)
@click.option(
    "--shared_size",
    "-s",
    "shared_size",
    multiple=True,
    type=int,
    help="size of shared axes of A and B used in muiltiplication",
)
@click.option(
    "--tile_size",
    "-t",
    "tile_size",
    multiple=True,
    type=int,
    help="size of tiles along 1 axes, used as [t, t, 1]",
)
@click.option(
    "--alpha",
    multiple=True,
    type=float,
    help="alpha, from C = beta * C + alpha * A@B",
    default=[1.0],
)
@click.option(
    "--beta",
    multiple=True,
    type=float,
    help="beta, from C = beta * C + alpha * A@B",
    default=[1.0],
)
@click.option(
    "--batches",
    "-r",
    "batches",
    multiple=True,
    type=int,
    help="butch size for all of the arrays",
)
def all_configurations(
    file, dtype, tA, tB, matrix_shape, shared_size, tile_size, alpha, beta, batches
):
    # Although batch can be implemented in several dimensions, only 1 is used
    # Don't know how to put into log otherwise
    batches = [[b] for b in batches]
    all_args = (
        dtype,
        tA,
        tB,
        matrix_shape,
        shared_size,
        tile_size,
        alpha,
        beta,
        batches,
    )
    if file.tell() == 0:
        file.write(
            "time, dtype, tA, tB, Cx, Cy, shared_size, tile_size, alpha, beta, batches\n"
        )
    for chosen_args in itertools.product(*all_args):
        str_type, str_tA, str_tB, *num_args = chosen_args
        nptype = {"float32": np.float32, "float64": np.float64}[str_type]
        tA = {"notrans": nntile.notrans, "trans": nntile.notrans}[str_tA]
        tB = {"notrans": nntile.notrans, "trans": nntile.notrans}[str_tB]
        iterations, timing = helper(nptype, tA, tB, *num_args)
        num_args = list(
            itertools.chain.from_iterable(
                [a if isinstance(a, tuple) else (a,) for a in num_args]
            )
        )
        num_args[-1] = num_args[-1][0]
        file.write(
            f"{timing / iterations}, {str_type}, {str_tA}, {str_tB}, "
            f"{', '.join(map(str, num_args))}\n"
        )


if __name__ == "__main__":
    all_configurations()
