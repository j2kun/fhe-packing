from circuit import CiphertextType, Value, Add, Mul, LeftRotate
from pprint import pprint


def test_repr():
    ct_ty = CiphertextType(num_slots=16)
    x = Value(name="input", ty=ct_ty)
    y = Add(x, x)
    z = Mul(y, x)
    w = LeftRotate(z, 3)
    print()
    pprint(w)
