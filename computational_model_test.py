from computational_model import Ciphertext, is_power_of_two, rotate_and_sum



def test_is_power_of_two():
    for i in range (10):
        assert is_power_of_two(1 << i)

    for i in range(2, 10):
        assert not is_power_of_two((1 << i) + 1)
        assert not is_power_of_two((1 << i) - 1)


def test_rotate_and_sum():
    x = Ciphertext(list(range(512)))
    y = rotate_and_sum(x)
    assert y.data == [sum(x.data)] * x.dim
