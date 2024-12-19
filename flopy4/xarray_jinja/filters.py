import sys

import numpy as np
import xarray as xr


def dask_expand(data: xr.DataArray):
    for block in data.data.to_delayed():
        block_data = block.compute()
        yield block_data


def nparray2string(data: np.ndarray):
    return np.array2string(
        data, separator=" ", precision=4, max_line_width=sys.maxsize
    )[1:-1]  # remove brackets
