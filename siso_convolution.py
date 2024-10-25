"""Jukevar-Vaikuntanathan-Chandrakasan SISO convolution technique from the Gazelle paper."""

from computational_model import Ciphertext
from computational_model import is_power_of_two
from util import (
    convolution_indices,
    flatten,
    map_matrix,
    pad_zeros,
    print_as_square,
    zeros,
)


def pack_rowwise(matrix):
    """Pack the matrix row-wise into a single ciphertext."""
    n = len(matrix)
    assert n == len(matrix[0])
    assert is_power_of_two(n)
    return Ciphertext(
        [matrix[i][j] for i in range(n) for j in range(n)], original_shape=(n, n)
    )


def prepare_filters(matrix_shape, filter, pad):
    """Construct punctured filters for SISO convolution."""
    n, m = matrix_shape
    fn, fm = len(filter), len(filter[0])
    filters = [[zeros(matrix_shape) for _ in range(fm)] for _ in range(fn)]

    indices = convolution_indices(matrix_shape=(n, m), filter_shape=(fn, fm), pad=pad, stride=1)
    for iter_index in indices:
        if iter_index.combined_index_within_bounds:
            fi, fj = iter_index.filter_index
            i, j = iter_index.combined_index
            filters[fi][fj][i][j] = filter[fi][fj]

    ciphertexts = map_matrix(
        filters, lambda f: Ciphertext(flatten(f), original_shape=(fn, fm))
    )

    for i in range(fn):
        for j in range(fm):
            # The Gazelle paper's rotation is backwards from our convention:
            # their positive rotation rotates index 0 leftward, while we rotate
            # rightward, so we need to negate the rotation amount given to our
            # rotation function.
            rotation = -m * (i - pad) - (j - pad)
            ciphertexts[i][j] = ciphertexts[i][j].rotate(rotation)
            print(f"Prepared filter for output entries using filter index ({i}, {j}), rotated by {rotation}:")
            print_as_square(ciphertexts[i][j])

    return ciphertexts


def siso_convolution(packed_matrix, matrix_shape, prepared_filters, pad=1):
    """Apply the SISO convolution to the packed matrix."""
    nrows, ncols = matrix_shape
    filter_width, filter_height = prepared_filters[0][0].original_shape

    output = Ciphertext(
        [0] * len(packed_matrix.data), original_shape=packed_matrix.original_shape
    )
    for i in range(filter_height):
        for j in range(filter_width):
            # The Gazelle paper's rotation is backwards from our convention:
            # their positive rotation rotates index 0 leftward, while we rotate
            # rightward, so we need to negate the rotation amount given to our
            # rotation function.
            rotation = -ncols * (i - pad) - (j - pad)
            rotated = packed_matrix.rotate(rotation)
            print(f"Rotated by {rotation} for filter index ({i}, {j}):")
            print_as_square(rotated)
            output += rotated * prepared_filters[i][j]

    return output


def plaintext_convolution(matrix, filter, pad):
    matrix = pad_zeros(matrix, pad)
    n, m = len(matrix), len(matrix[0])
    fn, fm = len(filter), len(filter[0])

    # naive implementation of convolution as loops
    expected = []
    for i in range(n):
        if i + fn > n:
            continue

        expected.append([])
        for j in range(m):
            if j + fm > m:
                continue

            accum = 0
            for fi in range(fn):
                for fj in range(fm):
                    accum += matrix[i + fi][j + fj] * filter[fi][fj]

            expected[-1].append(accum)

    return expected
