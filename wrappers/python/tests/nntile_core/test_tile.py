import nntile
import numpy as np
config = nntile.starpu.Config(1, 0, 0)

def test_tile_array():
    shape = [3, 4]
    traits = nntile.tile.TileTraits(shape)
    tile_fp32 = nntile.tile.Tile_fp32(traits)
    tile_fp64 = nntile.tile.Tile_fp64(traits)
    src_fp32 = np.array(np.random.randn(*shape), dtype=np.float32, order='F')
    src_fp64 = np.array(np.random.randn(*shape), dtype=np.float64, order='F')
    tile_fp32.from_array(src_fp32)
    tile_fp64.from_array(src_fp64)
    dst_fp32 = np.zeros(shape, dtype=np.float32, order='F')
    dst_fp64 = np.zeros(shape, dtype=np.float64, order='F')
    tile_fp32.to_array(dst_fp32)
    tile_fp64.to_array(dst_fp64)
    assert (dst_fp32 == src_fp32).all()
    assert (dst_fp64 == src_fp64).all()

