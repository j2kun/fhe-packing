import itertools
from fhelipe import Layout, GapBlock, AxisBitRange


def test_row_major_layout():
    tensor_shape = (4, 4)
    layout = Layout(
        [
            AxisBitRange.parse("d0[1:0]"),
            AxisBitRange.parse("d1[1:0]"),
        ]
    )
    actual = layout.expand(tensor_shape)
    expected = list(itertools.product(range(4), range(4)))
    assert actual == expected, actual


def test_column_major_layout():
    tensor_shape = (4, 4)
    layout = Layout(
        [
            AxisBitRange.parse("d1[1:0]"),
            AxisBitRange.parse("d0[1:0]"),
        ]
    )
    actual = layout.expand(tensor_shape)
    expected = [(i, j) for j in range(4) for i in range(4)]
    assert actual == expected, actual


def test_gap_example_from_fhelipe_paper_fig_3b():
    tensor_shape = (4,)
    layout = Layout(
        [
            AxisBitRange.parse("d0[1:0]"),
            GapBlock(2),
        ]
    )
    actual = layout.expand(tensor_shape)
    expected = [
        (0,), None, None, None,
        (1,), None, None, None,
        (2,), None, None, None,
        (3,), None, None, None,
    ]

    assert actual == expected, actual


# FIXME: can this support halevi-shoup?
# def test_halevi_shoup_diagonal_order():
#     tensor_shape = (4,4)
#     layout = Layout(
#         [
#             AxisBitRange.parse("d0[1:0]"),
#             AxisBitRange.parse("d1[1:0]"),
#         ]
#     )
#     actual = layout.expand(tensor_shape)
#     expected = [
#         (0,0), (1,1), (2,2), (3,3),
#         (0,1), (1,2), (2,3), (3,0),
#         (0,2), (1,3), (2,0), (3,1),
#         (0,3), (1,0), (2,1), (3,2),
#     ]
#
#    assert actual == expected, actual
