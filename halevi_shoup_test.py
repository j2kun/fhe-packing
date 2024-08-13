from computational_model import Ciphertext
from halevi_shoup import (
    pack,
    pack_naive,
    matrix_vector_multiply_naive,
    matrix_vector_multiply,
)


def test_naive_matmul():
    # each output entry ends up being 2*row[2] - 1
    matrix = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [6, 7, 8, 9]]
    vector = Ciphertext([1, -1, 2, 0])
    packed_matrix = pack_naive(matrix)
    result = matrix_vector_multiply_naive(packed_matrix, vector)
    assert result == Ciphertext([5, 9, 13, 15])


# def test_halevi_shoup_packing():
#     matrix = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
#     vector = Ciphertext([1, -1, 2])
#     packed_matrix = pack(matrix)
#     result = matrix_vector_multiply(packed_matrix, vector)
#     assert result == Ciphertext([5, 9, 13])
