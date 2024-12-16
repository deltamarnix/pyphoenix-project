import numpy as np
import xarray as xr


def dask_expand(data: xr.DataArray):
    for block in data.data.blocks:
        block_data = block.compute()
        yield block_data


def nparray2string(data: np.ndarray):
    return " ".join(data.astype(str))
