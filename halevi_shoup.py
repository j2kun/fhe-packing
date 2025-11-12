"""Halevi-Shoup matrix packing technique."""

from math import log2

from computational_model import Ciphertext
from computational_model import rotate_and_sum
from computational_model import is_power_of_two


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
    assert len(packed_matrix) == len(vector)

    n = len(packed_matrix)
    row_products = []
    for i in range(n):
        row_products.append(packed_matrix[i] * vector.rotate(-i))

    # Sum the results together
    result = row_products[0]
    for i in range(1, n):
        result += row_products[i]

    return result


def pack_squat(matrix: list[list[int]]) -> list[Ciphertext]:
    """Pack the matrix into a list of ciphertexts via
    Juvekar-Vaikuntanathan-Chandrakasan squat diagonal packing.
    """
    n, m = len(matrix), len(matrix[0])
    assert n < m
    assert is_power_of_two(n)
    assert is_power_of_two(m)

    ciphertexts = [[None] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            # wraps around the bottom of the matrix as well as the right.
            ciphertexts[i][j] = matrix[j % n][(i + j) % m]

    return [Ciphertext(ciphertexts[i]) for i in range(n)]


def matrix_vector_multiply_squat(
    packed_matrix: list[Ciphertext], vector: Ciphertext
) -> Ciphertext:
    """Multiply the squat-diagonal-packed matrix by the vector."""
    n, m = len(packed_matrix), len(packed_matrix[0])
    assert m == len(vector)
    assert n < m

    row_products = []
    for i in range(n):
        row_products.append(packed_matrix[i] * vector.rotate(-i))

    # Sum the results together
    partial_sums = row_products[0]
    for i in range(1, n):
        partial_sums += row_products[i]

    # Reduce the result to combine partial sums
    result = partial_sums
    num_shifts = int(log2(m) - log2(n))
    shift = m // 2
    for _ in range(num_shifts):
        result += result.rotate(shift)
        shift //= 2

    # Mask out the first n entries
    mask = [0] * m
    for i in range(n):
        mask[i] = 1

    return result * Ciphertext(mask)


if __name__ == "__main__":
    matrix = [[1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9, 10]]
    vector = [1, -1, 2, 0, 3, 4, -1, 0]
    packed_matrix = pack_squat(matrix)
    import numpy as np
    np_m = np.array(matrix)
    np_v = np.array(vector)
    np_result = np_m @ np_v
    print(np_result)
    import ipdb; ipdb.set_trace()
    result = matrix_vector_multiply_squat(packed_matrix, Ciphertext(vector))
