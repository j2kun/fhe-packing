from permutation_layout import Add
from permutation_layout import AffineExpr
from permutation_layout import AffineMap
from permutation_layout import Constant
from permutation_layout import DimId
from permutation_layout import Mul
from permutation_layout import column_major_layout
from permutation_layout import from_affine_map
from permutation_layout import row_major_layout


def test_single_ciphertext_row_major_layout():
    ciphertext_shape = (1, 32)
    data_shape = (4, 4)
    expected = """Layout[(1, 32) -> (4, 4)](
  (0, 0) -> (0, 0)
  (0, 1) -> (0, 1)
  (0, 2) -> (0, 2)
  (0, 3) -> (0, 3)
  (0, 4) -> (1, 0)
  (0, 5) -> (1, 1)
  (0, 6) -> (1, 2)
  (0, 7) -> (1, 3)
  (0, 8) -> (2, 0)
  (0, 9) -> (2, 1)
  (0, 10) -> (2, 2)
  (0, 11) -> (2, 3)
  (0, 12) -> (3, 0)
  (0, 13) -> (3, 1)
  (0, 14) -> (3, 2)
  (0, 15) -> (3, 3)
  (0, 16) -> G
  (0, 17) -> G
  (0, 18) -> G
  (0, 19) -> G
  (0, 20) -> G
  (0, 21) -> G
  (0, 22) -> G
  (0, 23) -> G
  (0, 24) -> G
  (0, 25) -> G
  (0, 26) -> G
  (0, 27) -> G
  (0, 28) -> G
  (0, 29) -> G
  (0, 30) -> G
  (0, 31) -> G)"""
    layout = row_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected

    affine_map = AffineMap(
        dims=["d0", "d1"],
        exprs=[DimId(dim_id="d0"), DimId(dim_id="d1")],
    )
    layout = from_affine_map(affine_map, data_shape, ciphertext_shape)
    assert str(layout) == expected


def test_multi_ciphertext_row_major_layout():
    ciphertext_shape = (2, 16)
    data_shape = (4, 6)
    expected = """Layout[(2, 16) -> (4, 6)](
  (0, 0) -> (0, 0)
  (0, 1) -> (0, 1)
  (0, 2) -> (0, 2)
  (0, 3) -> (0, 3)
  (0, 4) -> (0, 4)
  (0, 5) -> (0, 5)
  (0, 6) -> (1, 0)
  (0, 7) -> (1, 1)
  (0, 8) -> (1, 2)
  (0, 9) -> (1, 3)
  (0, 10) -> (1, 4)
  (0, 11) -> (1, 5)
  (0, 12) -> (2, 0)
  (0, 13) -> (2, 1)
  (0, 14) -> (2, 2)
  (0, 15) -> (2, 3)
  (1, 0) -> (2, 4)
  (1, 1) -> (2, 5)
  (1, 2) -> (3, 0)
  (1, 3) -> (3, 1)
  (1, 4) -> (3, 2)
  (1, 5) -> (3, 3)
  (1, 6) -> (3, 4)
  (1, 7) -> (3, 5)
  (1, 8) -> G
  (1, 9) -> G
  (1, 10) -> G
  (1, 11) -> G
  (1, 12) -> G
  (1, 13) -> G
  (1, 14) -> G
  (1, 15) -> G)"""

    layout = row_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected


def test_single_ciphertext_column_major_layout():
    ciphertext_shape = (1, 32)
    data_shape = (4, 4)
    expected = """Layout[(1, 32) -> (4, 4)](
  (0, 0) -> (0, 0)
  (0, 1) -> (1, 0)
  (0, 2) -> (2, 0)
  (0, 3) -> (3, 0)
  (0, 4) -> (0, 1)
  (0, 5) -> (1, 1)
  (0, 6) -> (2, 1)
  (0, 7) -> (3, 1)
  (0, 8) -> (0, 2)
  (0, 9) -> (1, 2)
  (0, 10) -> (2, 2)
  (0, 11) -> (3, 2)
  (0, 12) -> (0, 3)
  (0, 13) -> (1, 3)
  (0, 14) -> (2, 3)
  (0, 15) -> (3, 3)
  (0, 16) -> G
  (0, 17) -> G
  (0, 18) -> G
  (0, 19) -> G
  (0, 20) -> G
  (0, 21) -> G
  (0, 22) -> G
  (0, 23) -> G
  (0, 24) -> G
  (0, 25) -> G
  (0, 26) -> G
  (0, 27) -> G
  (0, 28) -> G
  (0, 29) -> G
  (0, 30) -> G
  (0, 31) -> G)"""
    layout = column_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected


def test_multi_ciphertext_column_major_layout():
    ciphertext_shape = (2, 16)
    data_shape = (4, 6)
    expected = """Layout[(2, 16) -> (4, 6)](
  (0, 0) -> (0, 0)
  (0, 1) -> (1, 0)
  (0, 2) -> (2, 0)
  (0, 3) -> (3, 0)
  (0, 4) -> (0, 1)
  (0, 5) -> (1, 1)
  (0, 6) -> (2, 1)
  (0, 7) -> (3, 1)
  (0, 8) -> (0, 2)
  (0, 9) -> (1, 2)
  (0, 10) -> (2, 2)
  (0, 11) -> (3, 2)
  (0, 12) -> (0, 3)
  (0, 13) -> (1, 3)
  (0, 14) -> (2, 3)
  (0, 15) -> (3, 3)
  (1, 0) -> (0, 4)
  (1, 1) -> (1, 4)
  (1, 2) -> (2, 4)
  (1, 3) -> (3, 4)
  (1, 4) -> (0, 5)
  (1, 5) -> (1, 5)
  (1, 6) -> (2, 5)
  (1, 7) -> (3, 5)
  (1, 8) -> G
  (1, 9) -> G
  (1, 10) -> G
  (1, 11) -> G
  (1, 12) -> G
  (1, 13) -> G
  (1, 14) -> G
  (1, 15) -> G)"""
    layout = column_major_layout(data_shape, ciphertext_shape)
    assert str(layout) == expected
