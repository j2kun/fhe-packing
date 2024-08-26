import pytest
from hypothesis import given
from hypothesis.strategies import composite, integers, lists

from computational_model import Ciphertext
from siso_convolution import (
    pack_rowwise,
    siso_convolution,
    plaintext_convolution,
)


@composite
def random_matrix(draw, shape=(4, 4)):
    """Generate a matrix of a given shape."""
    values = integers(min_value=-100, max_value=100)
    matrix_row_count, matrix_col_count = shape
    matrix = draw(
        lists(
            lists(values, min_size=matrix_col_count, max_size=matrix_col_count),
            min_size=matrix_row_count,
            max_size=matrix_row_count,
        ),
    )
    return matrix


def run_test(matrix, filter, pack_fn, conv_fn, pad=1):
    n, m = len(matrix), len(matrix[0])
    packed_matrix = pack_fn(matrix)
    result = conv_fn(packed_matrix, (n, m), filter, pad=pad)
    expected = plaintext_convolution(matrix, filter)
    expected = [x for row in expected for x in row]
    expected.extend([0] * (len(result.data) - len(expected)))
    assert result.data == expected, result.data


def test_plaintext_convolution():
    matrix = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
    filter = [[2, 0], [1, 2]]
    expected = [[10, 15, 20], [15, 20, 25], [20, 25, 30]]
    assert plaintext_convolution(matrix, filter, pad=0) == expected


def test_plaintext_convolution_padded():
    matrix = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
    filter = [[2, 0], [1, 2]]
    expected = [
        [2, 5, 8, 11, 4],
        [4, 10, 15, 20, 13],
        [6, 15, 20, 25, 16],
        [8, 20, 25, 30, 19],
        [0, 8, 10, 12, 14],
    ]
    assert plaintext_convolution(matrix, filter, pad=1) == expected


def test_simple():
    matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    filter = [[0, 0], [0, 1]]
    # expect 1 in index 8
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=0)


def test_simple2():
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    filter = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]
    import ipdb; ipdb.set_trace()
    # expect 1 in index 8
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=0)


@given(random_matrix(shape=(4, 4)), random_matrix(shape=(2, 2)))
def test_4_by_4_filter(matrix, filter):
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=0)
