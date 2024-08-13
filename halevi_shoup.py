"""Halevi-Shoup matrix packing technique."""

from computational_model import Ciphertext
from computational_model import rotate_and_sum


def pack_naive(matrix: list[list[int]]) -> list[Ciphertext]:
    """Naively pack the matrix into a list of ciphertexts."""
    assert len(matrix) == len(matrix[0])
    n = len(matrix)
    return [Ciphertext(matrix[i]) for i in range(n)]


def matrix_vector_multiply_naive(
    packed_matrix: list[Ciphertext], vector: Ciphertext
) -> Ciphertext:
    """Multiply the naively-packed matrix by the vector."""
    assert len(packed_matrix) == len(vector)

    n = len(packed_matrix)
    row_products = []
    for i in range(n):
        row_products.append(packed_matrix[i] * vector)

    # Each row_product needs to be sum-reduced
    reduced_row_products = []
    for row in row_products:
        reduced_row_products.append(rotate_and_sum(row))

    # Now we need to "select" the i-th entry of each reduced_row_product and
    # sum the extracted values together.

    extracted = []
    for i, row in enumerate(reduced_row_products):
        mask = [0] * n
        mask[i] = 1
        extracted.append(row * Ciphertext(mask))

    # Sum the masked values together
    result = extracted[0]
    for i in range(1, n):
        result += extracted[i]

    return result


def pack(matrix: list[list[int]]) -> list[Ciphertext]:
    """Pack the matrix into a list of ciphertexts via Halevi-Shoup."""
    assert len(matrix) == len(matrix[0])

    n = len(matrix)
    ciphertexts = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            ciphertexts[i][j] = matrix[j][(i + j) % n]

    return [Ciphertext(ciphertexts[i]) for i in range(n)]


def matrix_vector_multiply(
    packed_matrix: list[Ciphertext], vector: Ciphertext
) -> Ciphertext:
    """Multiply the Halevi-Shoup-packed matrix by the vector."""
    return None
