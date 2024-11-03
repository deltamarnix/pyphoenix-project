from os import linesep
from pathlib import Path

import numpy as np

from flopy4.mf6.io import MF6Transformer
from flopy4.mf6.io import make_parser as make_mf6_parser

COMPONENT = """
BEGIN OPTIONS
  K
  I 1
  D 1.0
  S hello world
  F FILEIN some/path
END OPTIONS

BEGIN PACKAGEDATA
  A INTERNAL 1.0 2.0 3.0
END PACKAGEDATA

BEGIN PERIOD 1
  FIRST
  FREQUENCY 2
END PERIOD 1

BEGIN PERIOD 2
  STEPS 1 2 3
END PERIOD 2
"""


MF6_PARSER = make_mf6_parser(
    params=["k", "i", "d", "s", "f", "a"],
    dict_blocks=["options", "packagedata"],
    list_blocks=["period"],
)
MF6_TRANSFORMER = MF6Transformer()


def test_parse_mf6():
    tree = MF6_PARSER.parse(COMPONENT)
    # view the parse tree with e.g.
    #   pytest test/test_lark.py::test_parse -s
    print(linesep + tree.pretty())


def test_transform_mf6():
    tree = MF6_PARSER.parse(COMPONENT)
    data = MF6_TRANSFORMER.transform(tree)
    assert data["options"] == {
        "d": 1.0,
        "f": Path("some/path"),
        "i": 1,
        "k": True,
        "s": "hello world",
    }
    assert np.array_equal(data["packagedata"]["a"], np.array([1.0, 2.0, 3.0]))
    assert data["period 1"][0] == ("FIRST",)
    assert data["period 1"][1] == ("FREQUENCY", 2)
    assert data["period 2"][0] == ("STEPS", 1, 2, 3)
