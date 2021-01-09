import pytest
from pytorch_nn_tools.train.progress import ProgressTracker


def test_progress_tracker__total_count():
    tracker = ProgressTracker()
    for i in tracker.track((1, 2, 3)):
        assert tracker.cnt_total_iter == i

    assert tracker.cnt_total_iter == 3, "counts must be preserved outside of the loop"
    for i in tracker.track((1, 2, 3, 4)):
        assert tracker.cnt_total_iter == i + 3, "counting must continue for the next sequence"

    assert tracker.cnt_total_iter == 7, "counts must be preserved outside of the loop"


def test_progress_tracker__count_in_epoch():
    tracker = ProgressTracker()
    for i in tracker.track((1, 2, 3)):
        assert tracker.cnt_iter_in_epoch == i
    assert tracker.cnt_iter_in_epoch == 3, "counts must be preserved outside of the loop"

    for i in tracker.track((1, 2, 3, 4)):
        assert tracker.cnt_iter_in_epoch == i
    assert tracker.cnt_iter_in_epoch == 4


def test_progress_tracker_preserves_len():
    tracker = ProgressTracker()
    with pytest.raises(ValueError):
        len(tracker)
    it = iter(tracker.track((1, 2, 3)))
    assert len(it) == 3
    next(it)
    assert len(it) == 3


def test_progress_tracker__doesnt_give_len_if_iterable_doesnt_have_it():
    tracker = ProgressTracker()
    with pytest.raises(ValueError):
        len(tracker)
    it = iter(tracker.track((i for i in range(3))))
    with pytest.raises(TypeError):
        len(it)

