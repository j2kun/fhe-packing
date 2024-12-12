from dataclasses import dataclass
import itertools


@dataclass(frozen=True, order=True)
class PermutationEntry:
    """A permutation entry is a mapping from a single ciphertext slot to a user
    data value.

    Let C be a list of ciphertexts, each of dimension N.
    Let D be a list of data vectors, each of dimension M.

    Then a single instance of this class represents one entry of a mapping

        C[ciphertext_index][slot_index] -> D[data_index][data_slot]
    """

    ciphertext_index: int
    ciphertext_slot: int
    data_index: int
    data_slot: int

    def __str__(self):
        return f"({self.ciphertext_index}, {self.ciphertext_slot}) -> ({self.data_index}, {self.data_slot})"


@dataclass(frozen=True, order=True)
class GapEntry:
    """A permutation entry marked as unused."""

    ciphertext_index: int
    ciphertext_slot: int

    def __str__(self):
        return f"({self.ciphertext_index}, {self.ciphertext_slot}) -> G"


class Layout:
    def __init__(self, ciphertext_shape, data_shape, entries):
        self.ciphertext_shape = ciphertext_shape
        self.data_shape = data_shape
        self.entries = list(
            sorted(entries, key=lambda x: (x.ciphertext_index, x.ciphertext_slot))
        )
        self.verify()

    def verify(self):
        num_ciphertexts, ciphertext_size = self.ciphertext_shape
        num_data, data_size = self.data_shape
        for entry in self.entries:
            assert entry.ciphertext_index < num_ciphertexts
            assert entry.ciphertext_slot < ciphertext_size
            if isinstance(entry, GapEntry):
                continue
            assert entry.data_index < num_data
            assert entry.data_slot < data_size

    def __str__(self):
        s = f"Layout[{self.ciphertext_shape} -> {self.data_shape}](\n"
        indent = "  "
        s += "\n".join([indent + str(entry) for entry in self.entries])
        s += ")"
        return s


def row_major_layout(data_shape, ciphertext_size):
    num_data, data_size = data_shape
    num_ciphertexts, ciphertext_size = ciphertext_size

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

        ci, cs = c
        di, ds = d
        entries.append(PermutationEntry(ci, cs, di, ds))

    return Layout(
        ciphertext_shape=(num_ciphertexts, ciphertext_size),
        data_shape=(num_data, data_size),
        entries=entries,
    )


def column_major_layout(data_shape, ciphertext_size):
    num_data, data_size = data_shape
    num_ciphertexts, ciphertext_size = ciphertext_size

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

        ci, cs = c
        di, ds = d
        entries.append(PermutationEntry(ci, cs, di, ds))

    return Layout(
        ciphertext_shape=(num_ciphertexts, ciphertext_size),
        data_shape=(num_data, data_size),
        entries=entries,
    )
