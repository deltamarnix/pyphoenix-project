import sys
import time
from typing import Any, Callable, Tuple

import dask.array as da
import jinja2
import numpy as np
import pytest
import xarray as xr
import xarray_extras
import xarray_extras.csv

import flopy4.xarray_jinja.filters as filters

test_combinations = [
    # (1_000, 100),
    # (1_000_000, 1_000),
    (1_000_000, 10_000),
    # (10_000_000, 1_000),
    (10_000_000, 10_000),
    # (100_000_000, 10_000),
    (100_000_000, 100_000),
    (100_000_000, 1_000_000),  # 1_000_000 is about 8MB of chunks.
    (
        100_000_000,
        10_000_000,
    ),  # 10_000_000 is about 80MB of chunks. Copilot advised 100MB.
]


@pytest.fixture(scope="module")
def memory_file():
    with open("memory.md", "w") as f:
        f.write("| test | args | memory (MB) | \n")
        f.write("| --- | --- | --- | \n")
        yield f


@pytest.fixture(scope="module")
def time_file():
    with open("times.md", "w") as f:
        f.write("| test | args | time (s) | \n")
        f.write("| --- | --- | --- | \n")
        yield f


def profile_function(
    func: Callable,
    args: Tuple[Any, ...],
    time_file,
    print_args={},
) -> None:
    start_time = time.time()
    func(*args)
    end_time = time.time()

    name = getattr(func, "__name__", "unknown")
    time_file.write(f"| {name} | {print_args} | {end_time - start_time} |\n")


def create_and_write_jinja(tmp_path, data: xr.DataArray):
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("flopy4.xarray_jinja"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["dask_expand"] = filters.dask_expand
    env.filters["nparray2string"] = filters.nparray2string

    generator = env.get_template("disu_template.disu.jinja").generate(
        data=data
    )
    with np.printoptions(precision=4, linewidth=sys.maxsize):
        with open(tmp_path / "test_xarray_to_text_jinja.disu", "w") as f:
            f.writelines(generator)


@pytest.mark.parametrize(
    "max_size,chunks",
    test_combinations,
)
@pytest.mark.skip("Too slow for large data")
def test_xarray_to_text_jinja(tmp_path, max_size, chunks, time_file):
    data = xr.DataArray(da.arange(0, max_size, 1), dims="x")
    data = data.chunk(chunks)
    profile_function(
        create_and_write_jinja,
        (tmp_path, data),
        time_file,
        print_args={"max_size": max_size, "chunks": chunks},
    )

    with open(tmp_path / "test_xarray_to_text_jinja.disu", "r") as f:
        output = f.readlines()
        assert (
            len(output) == 2 + max_size / chunks
        )  # begin + end + lines of data


def create_and_write_pandas(tmp_path, data: xr.DataArray):
    pandas_data = data.to_pandas()
    with open(tmp_path / "test_xarray_to_text_extras.disu", "w") as f:
        f.write("BEGIN GRIDDATA\n")
        pandas_data.to_csv(
            f,
            header=False,
            index=False,
            lineterminator=" ",
            float_format="%.4f",
        )
        f.write("\nEND GRIDDATA\n")


@pytest.mark.parametrize(
    "max_size,chunks",
    test_combinations,
)
def test_xarray_to_text_pandas(tmp_path, max_size, chunks, time_file):
    data = xr.DataArray(da.arange(0, max_size, 1), dims="x")
    data = data.chunk(chunks)
    profile_function(
        create_and_write_pandas,
        (tmp_path, data),
        time_file,
        print_args={"max_size": max_size, "chunks": chunks},
    )

    with open(tmp_path / "test_xarray_to_text_extras.disu", "r") as f:
        output = f.readlines()
        assert len(output) == 3  # begin + end + 1 line of data


def create_and_write_np_savetxt(tmp_path, data: xr.DataArray):
    with open(tmp_path / "test_xarray_to_text_raw.disu", "w") as f:
        f.write("BEGIN GRIDDATA\n")
        for block in data.data.to_delayed():
            block_data = block.compute()
            np.savetxt(f, block_data, newline=" ", fmt="%.4f")
        f.write("\nEND GRIDDATA\n")


@pytest.mark.parametrize(
    "max_size,chunks",
    test_combinations,
)
@pytest.mark.skip("Too slow for large data")
def test_xarray_to_text_np_savetxt(tmp_path, max_size, chunks, time_file):
    data = xr.DataArray(da.arange(0, max_size, 1), dims="x")
    data = data.chunk(chunks)
    profile_function(
        create_and_write_np_savetxt,
        (tmp_path, data),
        time_file,
        print_args={"max_size": max_size, "chunks": chunks},
    )

    with open(tmp_path / "test_xarray_to_text_raw.disu", "r") as f:
        output = f.readlines()
        assert len(output) == 3


def create_and_write_extras(tmp_path, data: xr.DataArray):
    file_path = tmp_path / "test_xarray_to_text_extras.disu"
    with open(file_path, "w") as f:
        f.write("BEGIN GRIDDATA\n")
    promise = xarray_extras.csv.to_csv(
        data,
        file_path,
        header=False,
        index=False,
        float_format="%.4f",
        lineterminator=" ",
        compression=None,
        mode="a",
    )
    promise.compute()
    # we have to open the file again,
    # because xarray_extras only allows paths and no file handlers.
    with open(file_path, "a") as f:
        f.write("\nEND GRIDDATA\n")


@pytest.mark.parametrize(
    "max_size,chunks",
    test_combinations,
)
def test_xarray_to_text_extras(tmp_path, max_size, chunks, time_file):
    data = xr.DataArray(da.arange(0, max_size, 1), dims="x")
    data = data.chunk(chunks)
    profile_function(
        create_and_write_extras,
        (tmp_path, data),
        time_file,
        print_args={"max_size": max_size, "chunks": chunks},
    )

    with open(tmp_path / "test_xarray_to_text_extras.disu", "r") as f:
        output = f.readlines()
        assert len(output) == 3
