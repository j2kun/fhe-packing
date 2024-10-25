import pytest
from hypothesis import given
from hypothesis.strategies import composite, integers, lists
from util import flatten

from siso_convolution import (
    pack_rowwise,
    siso_convolution,
    plaintext_convolution,
    prepare_filters,
)


def test_prepare_filters_with_pad():
    n = 4
    filter = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    pad = 1

    # fmt: off
    expected = [
        [
            [[0, 0, 0, 0],
             [0, 1, 1, 1],
             [0, 1, 1, 1],
             [0, 1, 1, 1]], # k1

            [[0, 0, 0, 0],
             [2, 2, 2, 2],
             [2, 2, 2, 2],
             [2, 2, 2, 2]], # k2

            [[0, 0, 0, 0],
             [3, 3, 3, 0],
             [3, 3, 3, 0],
             [3, 3, 3, 0]], # k3
        ],
        [
            [[0, 4, 4, 4],
             [0, 4, 4, 4],
             [0, 4, 4, 4],
             [0, 4, 4, 4]], # k4

            [[5, 5, 5, 5],
             [5, 5, 5, 5],
             [5, 5, 5, 5],
             [5, 5, 5, 5]], # k5

            [[6, 6, 6, 0],
             [6, 6, 6, 0],
             [6, 6, 6, 0],
             [6, 6, 6, 0]], # k6
        ],
        [
            [[0, 7, 7, 7],
             [0, 7, 7, 7],
             [0, 7, 7, 7],
             [0, 0, 0, 0]], # k7

            [[8, 8, 8, 8],
             [8, 8, 8, 8],
             [8, 8, 8, 8],
             [0, 0, 0, 0]], # k8

            [[9, 9, 9, 0],
             [9, 9, 9, 0],
             [9, 9, 9, 0],
             [0, 0, 0, 0]], # k9
        ],
    ]
    # fmt: on

    actual = prepare_filters((n, n), filter, pad)
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            assert actual[i][j].data == flatten(expected[i][j])


def test_prepare_filters_zero_pad():
    n = 4
    filter = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    pad = 0

    # fmt: off
    expected = [
        [
            [[1, 1, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k1

            [[2, 2, 0, 0],
             [2, 2, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k2

            [[3, 3, 0, 0],
             [3, 3, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k3
        ],
        [
            [[4, 4, 0, 0],
             [4, 4, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k4

            [[5, 5, 0, 0],
             [5, 5, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k5

            [[6, 6, 0, 0],
             [6, 6, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k6
        ],
        [
            [[7, 7, 0, 0],
             [7, 7, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k7

            [[8, 8, 0, 0],
             [8, 8, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k8

            [[9, 9, 0, 0],
             [9, 9, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], # k9
        ],
    ]
    # fmt: on

    actual = prepare_filters((n, n), filter, pad)
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            assert actual[i][j].data == flatten(expected[i][j])

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
    prepared_filters = prepare_filters((n, m), filter, pad=pad)
    result = conv_fn(packed_matrix, (n, m), prepared_filters, pad=pad)
    expected = plaintext_convolution(matrix, filter, pad=pad)
    expected = [x for row in expected for x in row]
    expected.extend([0] * (len(result.data) - len(expected)))
    assert expected == result.data


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


def test_example_from_article():
    matrix = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    filter = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=0)


@pytest.mark.parametrize("pad", [0, 1])
def test_simple(pad):
    matrix = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    filter = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # zero pad, so first entry should have a 1
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=pad)


def test_simple2():
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    filter = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=0)


def test_simple3():
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    filter = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=1)


@given(
    random_matrix(shape=(4, 4)),
    random_matrix(shape=(3, 3)),
)
def test_4_by_4_with_3_by_3_filter(matrix, filter):
    run_test(matrix, filter, pack_rowwise, siso_convolution, pad=1)


# @pytest.mark.parametrize("pad", list(range(10)))
# def test_larger_pads(pad):
#     matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
#     filter = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]
#     run_test(matrix, filter, pack_rowwise, siso_convolution, pad=pad)
