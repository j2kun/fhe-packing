from hypothesis import given, example
from hypothesis.strategies import composite, integers, lists
import math
import pytest

from bicyclic import (
    matrix_multiply,
    pack,
    unpack,
)

primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
prime_indices = integers(min_value=0, max_value=len(primes) - 1)


@composite
def coprime_matrix_dimensions(draw, max_size=7):
    """Generate coprime matrix dimensions (m, n) suitable for bicyclic encoding."""
    m = draw(prime_indices)
    n = draw(prime_indices.filter(lambda x: x != m))
    return primes[m], primes[n]


@composite
def coprime_triple_dimensions(draw, max_size=5):
    """Generate three pairwise coprime dimensions (m, n, p) for matrix multiplication."""
    m = draw(prime_indices)
    n = draw(prime_indices.filter(lambda x: x != m))
    p = draw(prime_indices.filter(lambda x: x != m and x != n))
    return primes[m], primes[n], primes[p]


@composite
def random_matrix(draw, m, n):
    """Generate an m×n matrix with random integer entries."""
    values = integers(min_value=-10, max_value=10)
    matrix = draw(
        lists(
            lists(values, min_size=n, max_size=n),
            min_size=m,
            max_size=m,
        )
    )
    return matrix


def test_bicyclic_encode_decode_basic():
    matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    encoded = pack(matrix, 15)
    decoded = unpack(encoded, 3, 5)
    assert decoded == matrix


def test_bicyclic_encode_example_from_paper():
    # Example from Section 4.1 of the paper
    matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    encoded = pack(matrix, 15)
    expected = [1, 7, 13, 4, 10, 11, 2, 8, 14, 5, 6, 12, 3, 9, 15]
    assert encoded.data == expected


def test_bicyclic_encode_non_coprime_fails():
    # 4x6 matrix where gcd(4,6) = 2 ≠ 1
    matrix = [[1, 2, 3, 4, 5, 6]] * 4

    with pytest.raises(ValueError, match="must be coprime"):
        pack(matrix, 24)


@given(coprime_matrix_dimensions())
def test_bicyclic_encode_decode_roundtrip(dims):
    m, n = dims
    matrix = [[i * n + j for j in range(n)] for i in range(m)]
    encoded = pack(matrix, m * n)
    decoded = unpack(encoded, m, n)
    assert decoded == matrix


def test_pack_replication():
    matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    expected_data = [1, 7, 13, 4, 10, 11, 2, 8, 14, 5, 6, 12, 3, 9, 15]
    expected_data += expected_data  # replicate to fill slots
    packed = pack(matrix, len(expected_data))
    assert packed.data == expected_data


def test_matrix_multiply_identity():
    A = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    B = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    num_slots = 3*5*7
    packed_A = pack(A, num_slots)
    packed_B = pack(B, num_slots)
    result = matrix_multiply(packed_A, packed_B, 3, 5, 7)
    decoded_result = unpack(result, 3, 7)
    expected = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ]
    assert decoded_result == expected


def naive_matrix_multiply(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, f"Inner dimensions must match: {n} != {n2}"

    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C


def test_matrix_multiply_example_from_paper():
    # Example from Section 4.1 of the paper
    A = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ]
    B = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
    ]
    num_slots = math.lcm(math.lcm(3, 5), 2)
    packA = pack(A, num_slots)
    packB = pack(B, num_slots)
    result = matrix_multiply(packA, packB, 3, 5, 2)
    decoded_result = unpack(result, 3, 2)
    expected = naive_matrix_multiply(A, B)
    assert decoded_result == expected


@given(coprime_triple_dimensions())
@example((3, 5, 2))
def test_matrix_multiply_random(dims):
    m, n, p = dims
    A = [[i * n + j for j in range(n)] for i in range(m)]
    B = [[i * p + j for j in range(p)] for i in range(n)]
    expected = naive_matrix_multiply(A, B)

    num_slots = math.lcm(math.lcm(m, n), p)
    packed_A = pack(A, num_slots)
    packed_B = pack(B, num_slots)

    result = matrix_multiply(packed_A, packed_B, m, n, p)
    decoded_result = unpack(result, m, p)
    assert decoded_result == expected


def test_free_transpose_property():
    matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    transpose = [[matrix[i][j] for i in range(3)] for j in range(5)]
    encoded_original = pack(matrix, 15)
    encoded_transpose = pack(transpose, 15)
    assert encoded_original == encoded_transpose
