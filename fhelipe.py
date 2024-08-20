"""An implementation of Fhelipe's layout notation abstraction."""
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import itertools


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n & (n - 1) == 0


@dataclass
class AxisBit:
    axis: int
    bit: int

    def __str__(self):
        return f"d{self.axis}[{self.bit}]"

    def __repr__(self):
        return self.__str__()


class LayoutEntry(ABC):
    @abstractmethod
    def entry_type(self) -> str:
        ...

    @abstractmethod
    def expand(self) -> list[Optional[AxisBit]]:
        ...

    @abstractmethod
    def __str__(self):
        ...

    def __repr__(self):
        return self.__str__()


class AxisBitRange(LayoutEntry):
    def __init__(self, axis: int = 0, start: int = 0, end: int = 0, stride: int = -1):
        """Create a range of bits on a given axis, in order from most
        significant bit to least.

        Args:
            axis: The axis to range over.
            start: The start bit index (inclusive).
            end: The end bit index (inclusive).
            stride: The stride between the start and end. Note the default value
              is -1 because most inputs will be in order from most significant to
              least significant bits.
        """
        if start >= end:
            assert stride < 0, f"{stride=} must be negative for decreasing ranges"
        elif start <= end:
            assert stride > 0, f"{stride=} must be positive for increasing ranges"
        self.axis = axis
        self.start = start
        self.end = end
        self.stride = stride

    @staticmethod
    def parse(string_repr):
        """Parse d{axis}[{start}:{end}] or d{axis}[{start}:{end}:{stride}]"""
        if string_repr.startswith("d"):
            axis, range_str = string_repr[1:].split("[")
            terms = range_str[:-1].split(":")
            if len(terms) not in [2, 3]:
                raise ValueError(f"Invalid string representation: {string_repr}")

            if len(terms) == 3:
                start, end, stride = terms
                return AxisBitRange(
                    axis=int(axis), start=int(start), end=int(end), stride=int(stride)
                )
            start, end = terms
            return AxisBitRange(axis=int(axis), start=int(start), end=int(end))
        raise ValueError(f"Invalid string representation: {string_repr}")

    def entry_type(self) -> str:
        return "BitRange"

    def expand(self):
        return [
            AxisBit(self.axis, bit)
            # add an extra self.stride for inclusive endpoints
            for bit in range(self.start, self.end + self.stride, self.stride)
        ]

    def __str__(self):
        prefix = f"d{self.axis}[{self.start}:{self.end}"
        if self.stride == 1:
            return f"{prefix}]"
        return f"{prefix}:{self.stride}]"


class GapBlock(LayoutEntry):
    def __init__(self, size: int = 1):
        assert size > 0
        self.size = size

    def entry_type(self) -> str:
        return "Gap"

    def expand(self):
        return [None] * self.size

    def __str__(self):
        if self.size == 1:
            return "(g)"
        return f"{self.size}*g"


class Layout:
    def __init__(self, entries: list[LayoutEntry]):
        self.entries = entries

    def expand(self, shape) -> list[Optional[int]]:
        """Expand the layout to a list of tuples of integers indexing a tensor.

        Gap values are expanded to None, while others are expanded to tuples of
        integers indexing a tensor of the given input shape.
        """
        for dim in shape:
            assert is_power_of_two(dim), f"{dim=} is not a power of two"

        bit_string = []
        for entry in self.entries:
            bit_string.extend(entry.expand())

        # Iterate over all indexes into the slots of the RLWE ciphertext.
        result = []
        last_non_gap_result = None
        for ciphertext_index in range(1 << len(bit_string)):
            # ciphertext_index is an index into an FHE ciphertext slot.
            # The bits of that index are interpreted according to the layout
            # defined by bit_string, i.e., each bit in ciphertext_index
            # corresponds to a bit of an index into the tensor that was packed
            # into the ciphertext. bit_string's entries describe how to
            # reconstruct the indices into the original tensor.
            indices = [0] * len(shape)
            for i, bit in enumerate(bit_string):
                if bit is None:
                    continue

                # the bit_string entries are ordered from most significant to
                # least significant bit, so the enumeration is in reverse order
                # of the bit we'd like to extract.
                i = len(bit_string) - i - 1
                extracted_bit = (ciphertext_index & (1 << i)) != 0

                # bit.bit describes how to interpret extracted_bit as a bit of
                # the index into the tensor.
                indices[bit.axis] |= (extracted_bit << bit.bit)

            # If the current index is the same as the previously stored
            # non-None index, then all the newly chnaged bits were gap bits.
            # This signifies striding, we add None to represent that.t
            tensor_index = tuple(indices)
            if last_non_gap_result is not None and tensor_index == last_non_gap_result:
                result.append(None)
            else:
                result.append(tensor_index)
                last_non_gap_result = tensor_index

        return result
