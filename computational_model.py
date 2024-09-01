"""The arithmetic/SIMD model of FHE."""


class Ciphertext:
    def __init__(self, data: list[int], original_shape: tuple[int, int] = None):
        self.data = data[:]
        self.dim = len(data)
        self.original_shape = original_shape

    def __len__(self) -> int:
        return self.dim

    def __eq__(self, other: "Ciphertext") -> bool:
        return self.data == other.data

    def __add__(self, other: "Ciphertext") -> "Ciphertext":
        assert self.dim == other.dim
        return Ciphertext([self.data[i] + other.data[i] for i in range(len(self.data))])

    def __mul__(self, other) -> "Ciphertext":
        if isinstance(other, Ciphertext):
            assert self.dim == other.dim
            return Ciphertext([self.data[i] * other.data[i] for i in range(len(self.data))])
        elif isinstance(other, list):
            # Plaintext-ciphertext multiplication
            assert self.dim == len(other) and isinstance(other[0], int)
            return Ciphertext([x * y for (x, y) in zip(self.data, other)])
        elif isinstance(other, int):
            # Plaintext-ciphertext multiplication
            return Ciphertext([other * x for x in self.data])

    def rotate(self, n: int) -> "Ciphertext":
        n = n % self.dim
        return Ciphertext(self.data[-n:] + self.data[:-n])

    def __repr__(self) -> str:
        return f"Ciphertext({self.data})"

    def __str__(self) -> str:
        return f"Ciphertext({self.data})"


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
