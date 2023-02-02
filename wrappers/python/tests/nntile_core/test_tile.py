# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tile.py
# Test for Tile<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-02

import nntile
import numpy as np
config = nntile.starpu.Config(1, 0, 0)

def helper(shape, Tile, dtype):
    traits = nntile.tile.TileTraits(shape)
    tile = Tile(traits)
    src = np.array(np.random.randn(*shape), dtype=dtype, order='F')
    dst = np.zeros_like(src)
    tile.from_array(src)
    tile.to_array(dst)
    nntile.starpu.wait_for_all()
    tile.unregister()
    return (dst == src).all()

def test():
    shape = [3, 4]
    assert helper(shape, nntile.tile.Tile_fp32, np.float32)
    assert helper(shape, nntile.tile.Tile_fp64, np.float64)

def test_repeat():
    shape = [3, 4]
    assert helper(shape, nntile.tile.Tile_fp32, np.float32)
    assert helper(shape, nntile.tile.Tile_fp64, np.float64)

if __name__ == "__main__":
    test()
    test_repeat()

