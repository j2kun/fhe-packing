from abc import ABC, abstractmethod
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


class Layout:

    def __init__(self, domain_shape, codomain_shape, entries, reversed=False):
        self.domain_shape = domain_shape
        self.codomain_shape = codomain_shape
        self.reversed = reversed

        def sort_key(x):
            return x.codomain_index if self.reversed else x.domain_index

        self.entries = list(sorted(entries, key=sort_key))
        # self.verify()

    def verify(self):
        for entry in self.entries:
            for index, size in zip(entry.domain_index, self.domain_shape):
                assert index < size, f"{index=} but {size=}"

            for index, size in zip(entry.codomain_index, self.codomain_shape):
                assert index < size, f"{index=} but {size=}"

    def __str__(self):
        if self.reversed:
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


def generate_iteration_space(dims):
    space = itertools.product(*[range(dim_size) for dim_size in dims])
    for indices in space:
        yield indices


def row_major_layout(data_shape, ciphertext_shape):
    num_data, data_size = data_shape
    num_ciphertexts, ciphertext_size = ciphertext_shape

    domain_entries = list(generate_iteration_space(data_shape))
    codomain_entries = list(generate_iteration_space(ciphertext_shape))
    entries = [
        PermutationEntry(c, d) for (c, d) in zip(domain_entries, codomain_entries)
    ]

    return Layout(
        domain_shape=data_shape,
        codomain_shape=ciphertext_shape,
        entries=entries,
    )


def column_major_layout(data_shape, ciphertext_shape):
    num_data, data_size = data_shape
    num_ciphertexts, ciphertext_size = ciphertext_shape

    domain_entries = []
    for data_slot in range(data_size):
        for data_index in range(num_data):
            domain_entries.append((data_index, data_slot))

    codomain_entries = list(generate_iteration_space(ciphertext_shape))
    entries = [
        PermutationEntry(c, d) for (c, d) in zip(domain_entries, codomain_entries)
    ]

    return Layout(
        domain_shape=data_shape,
        codomain_shape=ciphertext_shape,
        entries=entries,
    )


# Following MLIR's AffineExpr, we could also have Mod, FloorDiv, CeilDiv, but
# this suffices for demonstration.
class AffineExpr(ABC):

    def __init__(self, kind: str):
        self.kind = kind

    @abstractmethod
    def apply(self, env): ...


class Constant(AffineExpr):

    def __init__(self, value: int):
        super().__init__("const")
        self.value = value

    def apply(self, env):
        return self.value


class DimId(AffineExpr):

    def __init__(self, dim_id: str):
        super().__init__("dim_id")
        self.dim_id = dim_id

    def apply(self, env):
        return env[self.dim_id]


class Add(AffineExpr):

    def __init__(self, lhs: AffineExpr, rhs: AffineExpr):
        super().__init__("add")
        self.lhs = Constant(lhs) if isinstance(lhs, int) else lhs
        self.rhs = Constant(rhs) if isinstance(rhs, int) else rhs

    def apply(self, env):
        return self.lhs.apply(env) + self.rhs.apply(env)


class Mod(AffineExpr):

    def __init__(self, lhs: AffineExpr, rhs: AffineExpr):
        super().__init__("mod")
        self.lhs = lhs
        self.rhs = Constant(rhs) if isinstance(rhs, int) else rhs
        assert isinstance(self.rhs, Constant), "Affine mod op must have rhs constant!"

    def apply(self, env):
        return self.lhs.apply(env) % self.rhs.apply(env)


class FloorDiv(AffineExpr):

    def __init__(self, lhs: AffineExpr, rhs: AffineExpr):
        super().__init__("floordiv")
        self.lhs = lhs
        self.rhs = Constant(rhs) if isinstance(rhs, int) else rhs
        assert isinstance(
            self.rhs, Constant
        ), "Affine floordiv op must have rhs constant!"

    def apply(self, env):
        return self.lhs.apply(env) // self.rhs.apply(env)


class Mul(AffineExpr):

    def __init__(self, lhs: AffineExpr, rhs: AffineExpr):
        super().__init__("mul")
        self.lhs = Constant(lhs) if isinstance(lhs, int) else lhs
        self.rhs = Constant(rhs) if isinstance(rhs, int) else rhs
        assert isinstance(self.lhs, Constant) or isinstance(
            self.rhs, Constant
        ), "Affine mul op must have one operand constant!"

    def apply(self, env):
        return self.lhs.apply(env) * self.rhs.apply(env)


class AffineMap:
    def __init__(self, dims: list[str], exprs: list[AffineExpr]):
        self.dims = dims
        self.exprs = exprs

    def apply(self, env):
        for dim in self.dims:
            assert dim in env, f"Missing {dim} in env: {env}"
        return tuple(expr.apply(env) for expr in self.exprs)


def from_affine_map(affine_map: AffineMap, data_shape, ciphertext_shape):
    entries = []

    assert len(data_shape) == len(
        affine_map.dims
    ), f"Invalid shapes: {data_shape=} vs {len(affine_map.dims)=}"

    assert len(ciphertext_shape) == len(
        affine_map.exprs
    ), f"Invalid shapes: {ciphertext_shape=} vs {len(affine_map.exprs)=}"

    for c in generate_iteration_space(data_shape):
        env = dict(zip(affine_map.dims, c))
        entries.append(PermutationEntry(c, affine_map.apply(env)))

    return Layout(
        domain_shape=data_shape,
        codomain_shape=ciphertext_shape,
        entries=entries,
    )
