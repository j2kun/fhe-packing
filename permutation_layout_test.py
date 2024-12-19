from permutation_layout import Add
from permutation_layout import AffineExpr
from permutation_layout import AffineMap
from permutation_layout import Constant
from permutation_layout import DimId
from permutation_layout import FloorDiv
from permutation_layout import Mod
from permutation_layout import Mul
from permutation_layout import column_major_layout
from permutation_layout import from_affine_map
from permutation_layout import row_major_layout


def test_single_ciphertext_row_major_layout():
    ciphertext_shape = (1, 32)
    data_shape = (4, 4)
    expected = """Layout[(4, 4) -> (1, 32)](
  (0, 0) -> (0, 0)
  (0, 1) -> (0, 1)
  (0, 2) -> (0, 2)
  (0, 3) -> (0, 3)
  (1, 0) -> (0, 4)
  (1, 1) -> (0, 5)
  (1, 2) -> (0, 6)
  (1, 3) -> (0, 7)
  (2, 0) -> (0, 8)
  (2, 1) -> (0, 9)
  (2, 2) -> (0, 10)
  (2, 3) -> (0, 11)
  (3, 0) -> (0, 12)
  (3, 1) -> (0, 13)
  (3, 2) -> (0, 14)
  (3, 3) -> (0, 15))"""
    layout = row_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected

    affine_map = AffineMap(
        dims=["row", "col"],
        exprs=[
            Constant(0),
            Add(
                Mul(data_shape[1], DimId(dim_id="row")),
                DimId(dim_id="col"),
            ),
        ],
    )
    layout = from_affine_map(affine_map, data_shape, ciphertext_shape)
    assert str(layout) == expected


def test_multi_ciphertext_row_major_layout():
    ciphertext_shape = (2, 16)
    data_shape = (4, 6)
    expected = """Layout[(4, 6) -> (2, 16)](
  (0, 0) -> (0, 0)
  (0, 1) -> (0, 1)
  (0, 2) -> (0, 2)
  (0, 3) -> (0, 3)
  (0, 4) -> (0, 4)
  (0, 5) -> (0, 5)
  (1, 0) -> (0, 6)
  (1, 1) -> (0, 7)
  (1, 2) -> (0, 8)
  (1, 3) -> (0, 9)
  (1, 4) -> (0, 10)
  (1, 5) -> (0, 11)
  (2, 0) -> (0, 12)
  (2, 1) -> (0, 13)
  (2, 2) -> (0, 14)
  (2, 3) -> (0, 15)
  (2, 4) -> (1, 0)
  (2, 5) -> (1, 1)
  (3, 0) -> (1, 2)
  (3, 1) -> (1, 3)
  (3, 2) -> (1, 4)
  (3, 3) -> (1, 5)
  (3, 4) -> (1, 6)
  (3, 5) -> (1, 7))"""

    layout = row_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected

    index = Add(
        Mul(
            Constant(data_shape[1]),
            DimId(dim_id="row"),
        ),
        DimId(dim_id="col"),
    )
    affine_map = AffineMap(
        dims=["row", "col"],
        exprs=[
            FloorDiv(index, ciphertext_shape[1]),
            Mod(index, ciphertext_shape[1]),
        ],
    )
    layout = from_affine_map(affine_map, data_shape, ciphertext_shape)
    assert str(layout) == expected


def test_single_ciphertext_column_major_layout():
    ciphertext_shape = (1, 32)
    data_shape = (4, 4)
    expected = """Layout[(4, 4) -> (1, 32)](
  (0, 0) -> (0, 0)
  (0, 1) -> (0, 4)
  (0, 2) -> (0, 8)
  (0, 3) -> (0, 12)
  (1, 0) -> (0, 1)
  (1, 1) -> (0, 5)
  (1, 2) -> (0, 9)
  (1, 3) -> (0, 13)
  (2, 0) -> (0, 2)
  (2, 1) -> (0, 6)
  (2, 2) -> (0, 10)
  (2, 3) -> (0, 14)
  (3, 0) -> (0, 3)
  (3, 1) -> (0, 7)
  (3, 2) -> (0, 11)
  (3, 3) -> (0, 15))"""
    layout = column_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected

    index = Add(
        Mul(
            Constant(data_shape[0]),
            DimId(dim_id="col"),
        ),
        DimId(dim_id="row"),
    )
    affine_map = AffineMap(
        dims=["row", "col"],
        exprs=[
            Constant(0),
            index,
        ],
    )
    layout = from_affine_map(affine_map, data_shape, ciphertext_shape)
    assert str(layout) == expected


def test_multi_ciphertext_column_major_layout():
    ciphertext_shape = (2, 16)
    data_shape = (4, 6)
    expected = """Layout[(4, 6) -> (2, 16)](
  (0, 0) -> (0, 0)
  (0, 1) -> (0, 4)
  (0, 2) -> (0, 8)
  (0, 3) -> (0, 12)
  (0, 4) -> (1, 0)
  (0, 5) -> (1, 4)
  (1, 0) -> (0, 1)
  (1, 1) -> (0, 5)
  (1, 2) -> (0, 9)
  (1, 3) -> (0, 13)
  (1, 4) -> (1, 1)
  (1, 5) -> (1, 5)
  (2, 0) -> (0, 2)
  (2, 1) -> (0, 6)
  (2, 2) -> (0, 10)
  (2, 3) -> (0, 14)
  (2, 4) -> (1, 2)
  (2, 5) -> (1, 6)
  (3, 0) -> (0, 3)
  (3, 1) -> (0, 7)
  (3, 2) -> (0, 11)
  (3, 3) -> (0, 15)
  (3, 4) -> (1, 3)
  (3, 5) -> (1, 7))"""
    layout = column_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected

    index = Add(
        Mul(
            Constant(data_shape[0]),
            DimId(dim_id="col"),
        ),
        DimId(dim_id="row"),
    )
    affine_map = AffineMap(
        dims=["row", "col"],
        exprs=[
            FloorDiv(index, ciphertext_shape[1]),
            Mod(index, ciphertext_shape[1]),
        ],
    )
    layout = from_affine_map(affine_map, data_shape, ciphertext_shape)
    assert str(layout) == expected
