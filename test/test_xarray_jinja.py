import dask.array as da
import jinja2
import xarray as xr

import flopy4.xarray_jinja.filters as filters


def test_xarray_to_text_jinja(tmp_path):
    data = xr.DataArray(da.arange(0, 10_000, 1), dims="x")
    data = data.chunk(100)

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
    with open(tmp_path / "test_xarray_to_text_jinja.disu", "w") as f:
        f.writelines(generator)

    with open(tmp_path / "test_xarray_to_text_jinja.disu", "r") as f:
        output = f.readlines()
        assert len(output) == 102  # begin + end + 100 lines of data
