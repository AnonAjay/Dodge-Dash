"""Spec-by-test. These xfail until tiles_touched is implemented; then delete the marks."""
import pytest

from station_a.logic import tiles_touched

pending = pytest.mark.xfail(raises=NotImplementedError, strict=False)


@pending
def test_mid_tile_touches_one_tile():
    assert tiles_touched(0.3, 0.3, 0.25) == {(0, 0)}


@pending
def test_corner_touches_four_tiles():
    assert tiles_touched(0.6, 0.6, 0.25) == {(0, 0), (1, 0), (0, 1), (1, 1)}


@pending
def test_edge_touches_two_tiles():
    assert tiles_touched(0.6, 0.3, 0.25) == {(0, 0), (1, 0)}


@pending
def test_outside_grid_touches_nothing():
    assert tiles_touched(5.0, 5.0, 0.25) == set()
