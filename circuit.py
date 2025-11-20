"""An abstract circuit model on top of the computational model."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from computational_model import Ciphertext


class Type(ABC):
    @abstractmethod
    def __str__(self):
        ...


@dataclass(frozen=True)
class IntegerType(Type):
    def __str__(self):
        return "int"


@dataclass(frozen=True)
class CiphertextType(Type):
    num_slots: int

    def __str__(self):
        return f"ct[{self.num_slots}]"


@dataclass(frozen=True)
class PlaintextType(Type):
    num_slots: int

    def __str__(self):
        return f"pt[{self.num_slots}]"


@dataclass(frozen=True)
class TensorType(Type):
    shape: tuple[int]
    element_type: Type

    def __str__(self):
        shape_str = "x".join([str(x) for x in self.shape])
        return f"tensor<{shape_str}x{self.element_type}>"


class Node(ABC):
    pass


@dataclass(frozen=True)
class Value(Node):
    """An abstract value with a type."""
    name: str
    ty: Type


@dataclass(frozen=True)
class ConstantTensor(Node):
    value: Ciphertext


@dataclass(frozen=True)
class ConstantScalar(Node):
    value: int


@dataclass(frozen=True)
class Add(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Sub(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class Mul(Node):
    left: Node
    right: Node


@dataclass(frozen=True)
class LeftRotate(Node):
    """Cyclically rotate a ciphertext left; negative shift rotates right."""

    operand: Node
    shift: Node


@dataclass(frozen=True)
class Extract(Node):
    """Extract a single ciphertext from a list of ciphertexts."""

    operand: Node
    index: Node


@dataclass(frozen=True)
class FromElements(Node):
    """Combine ciphertexts into a list."""

    operands: tuple[Node]
