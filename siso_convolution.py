"""Jukevar-Vaikuntanathan-Chandrakasan SISO convolution technique."""

from computational_model import Ciphertext
from computational_model import is_power_of_two


def make_square(ciphertext):
    n = int(len(ciphertext.data) ** 0.5)
    for i in range(n):
        for j in range(n):
            print(f"{ciphertext.data[i*n + j]}, ", end="")
        print("")


def pack_rowwise(matrix):
    """Pack the matrix row-wise into a single ciphertext."""
    n = len(matrix)
    assert n == len(matrix[0])
    assert is_power_of_two(n)
    return Ciphertext([matrix[i][j] for i in range(n) for j in range(n)])


def prepare_filters(shape, filter):
    # FIXME: construct punctured filters
    pass


def siso_convolution(packed_matrix, image_shape, filters, pad=1):
    """Apply the SISO convolution to the packed matrix."""
    nrows, ncols = image_shape

    filter_width = len(filter)
    assert filter_width == len(filter[0])  # square filter

    output = Ciphertext([0] * len(packed_matrix.data))
    offset = (filter_width - pad) // 2
    for i in range(filter_width):
        for j in range(filter_width):
            rotation = ncols * (i + offset) + (j + offset)
            rotated = packed_matrix.rotate(rotation)
            print(make_square(rotated))
            output += rotated * filters[i][j]

    return output


def plaintext_convolution(matrix, filter, pad=1):
    matrix = [[0] * pad + row + [0] * pad for row in matrix]
    matrix = [[0] * len(matrix[0])] * pad + matrix + [[0] * len(matrix[0])] * pad

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
