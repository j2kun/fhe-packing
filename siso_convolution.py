"""Jukevar-Vaikuntanathan-Chandrakasan SISO convolution technique."""

from computational_model import Ciphertext
from computational_model import is_power_of_two
from util import flatten, zeros, ones, print_as_square, pad_zeros, map_matrix


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
    # for striding, you would divide by S before adding 1
    # output_shape = (n - fn + 2 * pad + 1, m - fm + 2 * pad + 1)
    test_matrix = ones(matrix_shape)
    test_matrix = pad_zeros(test_matrix, pad)
    padded_n, padded_m = len(test_matrix), len(test_matrix[0])

    filters = [[zeros(matrix_shape) for _ in range(fm)] for _ in range(fn)]
    # print("Test matrix:")
    # print_as_square(test_matrix)

    for i in range(padded_n):
        if i + fn > padded_n:
            continue
        for j in range(padded_m):
            if j + fm > padded_m:
                continue

            for fi in range(fn):
                for fj in range(fm):
                    if test_matrix[i + fi][j + fj] == 1:
                        filters[fi][fj][i][j] = filter[fi][fj]

    return map_matrix(filters, lambda f: Ciphertext(flatten(f), original_shape=(fn, fm)))


def siso_convolution(packed_matrix, image_shape, prepared_filters, pad=1):
    """Apply the SISO convolution to the packed matrix."""
    nrows, ncols = image_shape

    filter_width, filter_height = prepared_filters[0][0].original_shape
    assert filter_width == filter_height

    output = Ciphertext(
        [0] * len(packed_matrix.data), original_shape=packed_matrix.original_shape
    )
    offset = (filter_width - 1) // 2  # does this only work for odd-size filters?
    for i in range(filter_width):
        for j in range(filter_width):
            # The Gazelle paper is confusing because their rotation is
            # backwards: negative rotation rotates index 0 to the right, to a
            # larger index, so we need to negative the rotation amount given to
            # our rotation function.
            rotation = ncols * (i - offset) + (j - offset)
            rotated = packed_matrix.rotate(-rotation)
            print(print_as_square(rotated))
            output += rotated * prepared_filters[i][j]

    return output


def plaintext_convolution(matrix, filter, pad=1):
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
