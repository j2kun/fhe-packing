import itertools
from util import convolution_indices, flatten, ConvolutionIterationIndex


def test_convolution_indices():
    base = list(itertools.product(list(range(4)), list(range(4))))
    expected_base = flatten([list(itertools.repeat(t, 4)) for t in base])

    filter = list(itertools.product(list(range(2)), list(range(2))))
    expected_filter = flatten(itertools.repeat(filter, 16))

    actual = list(convolution_indices((4, 4), (2, 2)))
    actual_base = [x.base_index for x in actual]
    actual_filter = [x.filter_index for x in actual]

    assert actual_base == expected_base
    assert actual_filter == expected_filter


def test_convolution_indices_pad():
    actual = list(convolution_indices((2, 2), (2, 2), pad=2))
    assert actual[0] == ConvolutionIterationIndex(
        base_index=(-2, -2),
        filter_index=(0, 0),
        base_index_within_bounds=False,
        filter_index_within_bounds=False,
    )
    assert actual[16] == ConvolutionIterationIndex(
        base_index=(-2, 2),
        filter_index=(0, 0),
        base_index_within_bounds=False,
        filter_index_within_bounds=False,
    )
    assert actual[38] == ConvolutionIterationIndex(
        base_index=(-1, 1),
        filter_index=(1, 0),
        base_index_within_bounds=False,
        filter_index_within_bounds=True,
    )
    assert actual[56] == ConvolutionIterationIndex(
        base_index=(0, 0),
        filter_index=(0, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )
    assert actual[58] == ConvolutionIterationIndex(
        base_index=(0, 0),
        filter_index=(1, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )


def test_convolution_indices_lopsided_pad():
    actual = list(convolution_indices((2, 2), (2, 2), pad=(1, 2)))
    assert actual[0] == ConvolutionIterationIndex(
        base_index=(-1, -2),
        filter_index=(0, 0),
        base_index_within_bounds=False,
        filter_index_within_bounds=False,
    )
    assert actual[20] == ConvolutionIterationIndex(
        base_index=(-1, 3),
        filter_index=(0, 0),
        base_index_within_bounds=False,
        filter_index_within_bounds=False,
    )
    assert actual[24] == ConvolutionIterationIndex(
        base_index=(0, -2),
        filter_index=(0, 0),
        base_index_within_bounds=False,
        filter_index_within_bounds=False,
    )


def test_convolution_indices_stride():
    actual = list(convolution_indices((4, 4), (2, 2), stride=2))
    assert actual[0] == ConvolutionIterationIndex(
        base_index=(0, 0),
        filter_index=(0, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )
    assert actual[4] == ConvolutionIterationIndex(
        base_index=(0, 2),
        filter_index=(0, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )
    assert actual[8] == ConvolutionIterationIndex(
        base_index=(2, 0),
        filter_index=(0, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )


def test_convolution_indices_lopsided_stride():
    actual = list(convolution_indices((4, 4), (2, 2), stride=(1, 2)))
    assert actual[0] == ConvolutionIterationIndex(
        base_index=(0, 0),
        filter_index=(0, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )
    assert actual[4] == ConvolutionIterationIndex(
        base_index=(0, 2),
        filter_index=(0, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )
    assert actual[8] == ConvolutionIterationIndex(
        base_index=(1, 0),
        filter_index=(0, 0),
        base_index_within_bounds=True,
        filter_index_within_bounds=True,
    )
