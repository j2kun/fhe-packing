import itertools
from dataclasses import dataclass
from computational_model import Ciphertext


def flatten(matrix):
    return [x for row in matrix for x in row]


def fill(shape, value):
    """Create a matrix of value with the given shape."""
    return [[value] * shape[1] for _ in range(shape[0])]


def zeros(shape):
    """Create a matrix of zeros with the given shape."""
    return fill(shape, 0)


def ones(shape):
    """Create a matrix of ones with the given shape."""
    return fill(shape, 1)


def as_square(matrix):
    data = matrix.data if isinstance(matrix, Ciphertext) else matrix
    if isinstance(data[0], list):
        data = flatten(data)
    n = int(len(data) ** 0.5)
    s = ""
    for i in range(n):
        for j in range(n):
            s += f"{data[i*n + j]}, "
        s += "\n"
    return s


def print_as_square(matrix):
    print(as_square(matrix))


def pad_zeros(matrix, pad):
    """Pad the matrix with zeros."""
    padding_row = [0] * (len(matrix[0]) + 2 * pad)
    matrix = [[0] * pad + row + [0] * pad for row in matrix]
    matrix = (
        [padding_row[:] for _ in range(pad)]
        + matrix
        + [padding_row[:] for _ in range(pad)]
    )
    return matrix


def map_matrix(matrix, fn):
    return [[fn(x) for x in row] for row in matrix]


@dataclass(frozen=True)
class ConvolutionIterationIndex:
    """An index for iterating over the convolution of a matrix and a filter.
    Returned by convolution_indices."""

    # The base_index tracks the current position of the top left of the
    # filter as it slides along the matrix. (0, 0) is the top left of the
    # matrix, so when padding is nonzero, these indices can be negative.
    base_index: tuple[int]

    # The filter_index tracks the current position within the filter.
    # i.e., for a (3, 3) filter, the filter indices will range from 0 to 2
    # every 9 yields from convolution_indices.
    filter_index: tuple[int]

    # the combined index is the base_index + filter_index
    combined_index: tuple[int]

    # true or false depending on whether the current index position is within
    # the bounds of the original (unpadded) matrix.
    base_index_within_bounds: bool
    combined_index_within_bounds: bool


def convolution_indices(matrix_shape, filter_shape, pad=0, stride=1):
    """Generate indices for the convolution of a matrix and a filter.

    matrix_shape: the per-axis dimensions of the matrix
    filter_shape: the per-axis dimensions of the filter
    pad: the per-axis amount of padding (added to the beginning and end of each axis)
    stride: the per-axis stride
    """
    if type(pad) == int:
        pad = [pad] * len(matrix_shape)
    if type(stride) == int:
        stride = [stride] * len(matrix_shape)

    assert len(pad) == len(matrix_shape) == len(filter_shape) == len(stride)

    start = [-x for x in pad]
    stop = [x + y - f + 1 for x, y, f in zip(matrix_shape, pad, filter_shape)]

    matrix_iter_ranges = [
        list(range(start, stop, stride))
        for start, stop, stride in zip(start, stop, stride)
    ]
    filter_iter_ranges = [list(range(0, dim)) for dim in filter_shape]

    for indices in itertools.product(*matrix_iter_ranges):
        base_index = tuple(indices)
        base_index_within_bounds = all(
            0 <= ndx < dim for ndx, dim in zip(base_index, matrix_shape)
        )

        for filter_index in itertools.product(*filter_iter_ranges):
            combined_index_within_bounds = all(
                0 <= base + f_index < dim
                for base, f_index, dim in zip(base_index, filter_index, matrix_shape)
            )
            yield ConvolutionIterationIndex(
                base_index=base_index,
                filter_index=filter_index,
                combined_index=tuple(b + f for b, f in zip(base_index, filter_index)),
                base_index_within_bounds=base_index_within_bounds,
                combined_index_within_bounds=combined_index_within_bounds,
            )
