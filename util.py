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
