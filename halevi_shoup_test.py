import pytest

from computational_model import Ciphertext
from halevi_shoup import (
    pack,
    pack_naive,
    matrix_vector_multiply_naive,
    matrix_vector_multiply,
)


@pytest.mark.parametrize(
    "pack_fn,mul_fn",
    [(pack_naive, matrix_vector_multiply_naive), (pack, matrix_vector_multiply)],
)
def test_matmul(pack_fn, mul_fn):
    # each output entry ends up being 2*row[2] - 1
    matrix = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [6, 7, 8, 9]]
    vector = Ciphertext([1, -1, 2, 0])
    packed_matrix = pack_fn(matrix)
    result = mul_fn(packed_matrix, vector)
    assert result == Ciphertext([5, 9, 13, 15]), result.data
