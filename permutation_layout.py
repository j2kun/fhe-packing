from dataclasses import dataclass
import itertools


@dataclass(frozen=True, order=True)
class PermutationEntry:
    """A permutation entry is a mapping from a domain to a codomain.

    Let C be a list of ciphertexts, each of dimension N.
    Let D be a list of data vectors, each of dimension M.

    Then a single instance of this class represents one entry of a mapping

        C[ciphertext_index][slot_index] -> D[data_index][data_slot]

    Or it could represent a mapping

        D[data_index][data_slot] -> C[ciphertext_index][slot_index]

    The two options are because I'm exploring the possibility of supporting
    both, or determining that one direction is better than the other.
    """

    domain_index: tuple[int]
    codomain_index: tuple[int]

    def __str__(self):
        return f"{self.domain_index} -> {self.codomain_index}"

    def reversed_str(self):
        return f"{self.codomain_index} -> {self.domain_index}"


@dataclass(frozen=True, order=True)
class GapEntry:
    """A permutation entry marked as unused."""

    index: tuple[int]

    def __str__(self):
        return f"({self.index} -> G"


class Layout:

    def __init__(self, domain_shape, codomain_shape, entries, reverse=False):
        self.domain_shape = domain_shape
        self.codomain_shape = codomain_shape

        def sort_key(x):
            return x.codomain_index if reverse else x.domain_index

        self.entries = list(sorted(entries, key=sort_key))
        self.verify()

    def verify(self):
        for entry in self.entries:
            for index, size in zip(entry.domain_index, self.domain_shape):
                assert index < size

            if isinstance(entry, GapEntry):
                continue

            for index, size in zip(entry.codomain_index, self.codomain_shape):
                assert index < size

    def __str__(self):
        if reversed:
            s = f"Layout[{self.codomain_shape} -> {self.domain_shape}](\n"
            indent = "  "
            s += "\n".join([indent + entry.reversed_str() for entry in self.entries])
            s += ")"
            return s

        s = f"Layout[{self.domain_shape} -> {self.codomain_shape}](\n"
        indent = "  "
        s += "\n".join([indent + str(entry) for entry in self.entries])
        s += ")"
        return s


def row_major_layout(data_shape, ciphertext_shape):
    num_data, data_size = data_shape
    num_ciphertexts, ciphertext_size = ciphertext_shape

    entries = []
    domain_entries = []
    codomain_entries = []
    for ciphertext_index in range(num_ciphertexts):
        for slot_index in range(ciphertext_size):
            domain_entries.append((ciphertext_index, slot_index))

    for data_index in range(num_data):
        for data_slot in range(data_size):
            codomain_entries.append((data_index, data_slot))

    for c, d in itertools.zip_longest(domain_entries, codomain_entries):
        if not d:
            entries.append(GapEntry(*c))
            continue

        entries.append(PermutationEntry(c, d))

    return Layout(
        ciphertext_shape=ciphertext_shape,
        data_shape=(num_data, data_size),
        entries=entries,
    )


def column_major_layout(data_shape, ciphertext_shape):
    num_data, data_size = data_shape
    num_ciphertexts, ciphertext_size = ciphertext_shape

    entries = []
    domain_entries = []
    codomain_entries = []
    for ciphertext_index in range(num_ciphertexts):
        for slot_index in range(ciphertext_size):
            domain_entries.append((ciphertext_index, slot_index))

    for data_slot in range(data_size):
        for data_index in range(num_data):
            codomain_entries.append((data_index, data_slot))

    for c, d in itertools.zip_longest(domain_entries, codomain_entries):
        if not d:
            entries.append(GapEntry(*c))
            continue

        entries.append(PermutationEntry(c, d))

    return Layout(
        ciphertext_shape=cipertext_shape,
        data_shape=(num_data, data_size),
        entries=entries,
    )


# Following MLIR's AffineExpr, we could also have Mod, FloorDiv, CeilDiv, but
# this suffices for demonstration.
class AffineExpr:

    def __init__(self, kind: str):
        self.kind = kind


class Constant(AffineExpr):

    def __init__(self, value: int):
        super().__init__(self, "const")
        self.value = value


class DimId(AffineExpr):

    def __init__(self, dim_id: str):
        super().__init__(self, "dim_id")
        self.dim_id = dim_id


class Add(AffineExpr):

    def __init__(self, lhs: AffineExpr, rhs: AffineExpr):
        super().__init__(self, "add")
        self.lhs = lhs
        self.rhs = rhs


class Mul(AffineExpr):

    def __init__(self, lhs: AffineExpr, rhs: AffineExpr):
        super().__init__(self, "mul")
        self.lhs = lhs
        self.rhs = rhs


class AffineMap:
    def __init__(self, dims: list[str], exprs: list[AffineExpr]):
        self.dims = dims
        self.exprs = exprs


def from_affine_map(affine_map: AffineMap, data_shape, ciphertext_shape):
    # The domain is the data iteration space, and the codomain
    # is the ciphertext iteration space
    data_iteration_space = itertools.product(
        [range(dim_size) for dim_size in data_shape]
    )

    entries = []
    domain_entries = []
    codomain_entries = []
    for indices in domain_iteration_space:
        domain_entries.append(indices)

    for data_slot in range(data_size):
        for data_index in range(num_data):
            codomain_entries.append((data_index, data_slot))

    for c, d in itertools.zip_longest(domain_entries, codomain_entries):
        if not d:
            entries.append(GapEntry(*c))
            continue

        ci, cs = c
        di, ds = d
        entries.append(PermutationEntry(ci, cs, di, ds))

    return Layout(
        ciphertext_shape=cipertext_shape,
        data_shape=(num_data, data_size),
        entries=entries,
    )
