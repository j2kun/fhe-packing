"""Bicyclic matrix multiplication encoding.

Algorithm 1 from
Homomorphic Matrix Operations under Bicyclic Encoding
Jingwei Chen, Linhan Yang, Wenyuan Wu, Yang Liu, Yong Feng
https://eprint.iacr.org/2024/1762
"""

from math import gcd, ceil

from computational_model import Ciphertext


def pairwise_coprime(*args: int) -> bool:
    """Check if all provided integers are pairwise coprime."""
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            if gcd(args[i], args[j]) != 1:
                return False
    return True


def pack(matrix: list[list[int]], num_slots: int) -> Ciphertext:
    """
    Encode a matrix using bicyclic encoding.

    For an m×n matrix A, the bicyclic encoding φ(A) is defined as:
    φ(A)_k = a_{k mod m, k mod n} for all k ∈ Z_{mn}

    This requires gcd(m, n) = 1 for the encoding to be invertible.
    """
    rows, cols = len(matrix), len(matrix[0])
    if gcd(rows, cols) != 1:
        raise ValueError(
            f"Matrix dimensions must be coprime. Got gcd({rows}, {cols}) = {gcd(rows, cols)}"
        )

    data = [matrix[k % rows][k % cols] for k in range(rows * cols)]
    while len(data) < num_slots:
        # duplicate the data until it fills the slots
        data += data
    # truncate to the exact slot size
    data = data[:num_slots]

    return Ciphertext(data)


def unpack(encoded: Ciphertext, m: int, n: int) -> list[list[int]]:
    """
    Decode a bicyclically encoded vector back to a matrix.

    Uses the Chinese Remainder Theorem to recover the original matrix.
    """
    if gcd(m, n) != 1:
        raise ValueError(
            f"Matrix dimensions must be coprime. Got gcd({m}, {n}) = {gcd(m, n)}"
        )

    matrix = [[0] * n for _ in range(m)]

    # Using Chinese Remainder Theorem to decode
    n_inv_mod_m = pow(n, -1, m)  # n^{-1} mod m
    m_inv_mod_n = pow(m, -1, n)  # m^{-1} mod n

    for i in range(m):
        for j in range(n):
            k = (i * n * n_inv_mod_m + j * m * m_inv_mod_n) % (m * n)
            matrix[i][j] = encoded.data[k]

    return matrix


def matrix_multiply(
    packed_matrix_a: Ciphertext, packed_matrix_b: Ciphertext, m: int, n: int, p: int
) -> Ciphertext:
    """
    Multiply two bicyclically encoded matrices A (m×n) and B (n×p) to get C (m×p).

    This implements the bicyclic matrix multiplication algorithm BMM-I from the paper.
    The result uses optimal multiplicative depth of 1.

    Requires gcd(m, n, p) = 1 for correctness.
    """
    assert pairwise_coprime(m, n, p), "Dimensions must be pairwise coprime."
    assert len(packed_matrix_a) == len(
        packed_matrix_b
    ), "Both ciphertexts must have the same number of slots"

    result = Ciphertext([0] * len(packed_matrix_a))
    print(f"m={m}, n={n}, p={p}")

    r = ceil(n / m)
    while (r*n - m) % p != 0:
        r += 1
    print(f"Using r = {r} for BMM-I")

    for i in range(n):
        a_rot = (-i * m) % (m*n)
        b_rot = (i * (r*n - m)) % (n*p)
        rotated_a = packed_matrix_a.rotate(a_rot)
        rotated_b = packed_matrix_b.rotate(b_rot)
        prod = (rotated_a * rotated_b)
        print(f"step={i}, i_a={a_rot}, i_b={b_rot}, rotated_a={rotated_a}, rotated_b={rotated_b}, prod={prod}")
        result += prod

    return result
