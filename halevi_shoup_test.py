import pytest
from hypothesis import given
from hypothesis.strategies import composite, integers, lists

from computational_model import Ciphertext
from halevi_shoup import (
    pack,
    pack_naive,
    pack_squat,
    matrix_vector_multiply_naive,
    matrix_vector_multiply,
    matrix_vector_multiply_squat,
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


@composite
def random_vector(draw, dim=4):
    """Generate a vector of a given shape."""
    values = integers(min_value=-100, max_value=100)
    matrix = draw(lists(values, min_size=dim, max_size=dim))
    return matrix


def run_test(matrix, vector, pack_fn, mul_fn):
    packed_matrix = pack_fn(matrix)
    result = mul_fn(packed_matrix, Ciphertext(vector))

    expected = [0] * len(matrix)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            expected[i] += val * vector[j]

    # only relevant for squat diagonal packing
    expected = expected + [0] * (len(matrix[0]) - len(expected))
    assert result.data == expected, result.data


@pytest.mark.parametrize(
    "pack_fn,mul_fn",
    [(pack_naive, matrix_vector_multiply_naive), (pack, matrix_vector_multiply)],
)
def test_matmul(pack_fn, mul_fn):
    matrix = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [6, 7, 8, 9]]
    vector = [1, -1, 2, 0]
    run_test(matrix, vector, pack_fn, mul_fn)


for n in [2, 4, 8, 16]:

    @given(random_matrix(shape=(n, 2 * n)), random_vector(dim=2 * n))
    def test_matmul_squat(matrix, vector):
        run_test(matrix, vector, pack_squat, matrix_vector_multiply_squat)
