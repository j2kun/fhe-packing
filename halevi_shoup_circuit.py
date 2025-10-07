"""Halevi-Shoup matrix packing technique."""

from typing import Callable

from computational_model import Ciphertext
from circuit import (
    Add,
    CiphertextType,
    ConstantScalar,
    ConstantTensorNode,
    Extract,
    FromElements,
    IntegerType,
    LeftRotate,
    Mul,
    Node,
    PlaintextType,
    TensorType,
    Value,
)


KernelFn = Callable[[Value, Value], Node]
MatrixPackFn = Callable[[list[list[int]]], list[Ciphertext]]
VectorPackFn = Callable[[list[int]], list[Ciphertext]]


def make_matrix_packing_func(tensor_type: TensorType) -> MatrixPackFn:
    """Return a packing function for the cleartext matrix of a matvec."""

    def pack_matrix(matrix: list[list[int]]) -> list[Ciphertext]:
        n = len(matrix)
        ciphertexts = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ciphertexts[i][j] = matrix[j][(i + j) % n]
        return [Ciphertext(ciphertexts[i]) for i in range(n)]

    return pack_matrix


def make_vector_packing_func(tensor_type: TensorType) -> VectorPackFn:
    """Return a packing function for the ciphertext vector of a matvec."""

    def pack_vector(vector: list[int]) -> list[Ciphertext]:
        n = len(vector)
        ciphertext = [0] * n
        for i in range(n):
            ciphertext[i] = vector[i % n]
        return [Ciphertext(ciphertext)]

    return pack_vector


def make_matvec_kernel(matrix_ty: TensorType) -> KernelFn:
    """Define an FHE kernel that implements matrix-vector multiplication.

    Args:
        matrix_ty: the ranked tensor type of the original data matrix
          (axis 0 = rows, axis 1 = cols).

    Returns:
        A callable implementing the matvec kernel as a circuit when given
        two inputs:

            - packed_matrix: the packed matrix SSA value as a rank 1 tensor of
              plaintexts
            - packed_vector: the packed matrix SSA value as a rank 1 tensor of
              ciphertexts
    """
    num_rows = matrix_ty.shape[0]

    def kernel(packed_matrix: Value, packed_vector: Value) -> Node:
        row_products = []
        for i in range(num_rows):
            vec_rotated = LeftRotate(packed_vector, i)
            diagonal_i = Extract(packed_matrix, i)
            row_products.append(Mul(diagonal_i, vec_rotated))

        result = row_products[0]
        for i in range(1, num_rows):
            result = Add(result, row_products[i])

        return result

    return kernel


# TODO: implement scoring function
# TODO: implement evaluator
