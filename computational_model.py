"""The arithmetic/SIMD model of FHE."""


class Ciphertext:
    def __init__(self, data: list[int]):
        self.data = data[:]
        self.dim = len(data)

    def __add__(self, other: "Ciphertext") -> "Ciphertext":
        assert self.dim == other.dim
        return Ciphertext([self.data[i] + other.data[i] for i in range(len(self.data))])

    def __mul__(self, other: "Ciphertext") -> "Ciphertext":
        assert self.dim == other.dim
        return Ciphertext([self.data[i] * other.data[i] for i in range(len(self.data))])

    def rotate(self, n: int) -> "Ciphertext":
        n = n % self.dim
        return Ciphertext(self.data[-n:] + self.data[:-n])


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n & (n - 1) == 0


def rotate_and_sum(ciphertext: Ciphertext) -> Ciphertext:
    """Return a ciphertext where each entry contains the sum of all entries in
    the input ciphertext."""
    n = len(ciphertext.data)
    assert is_power_of_two(n)
    # copy so as not to mutate the input
    result = Ciphertext(ciphertext.data[:])

    shift = n // 2
    while shift > 0:
        result += result.rotate(shift)
        shift //= 2

    return result
