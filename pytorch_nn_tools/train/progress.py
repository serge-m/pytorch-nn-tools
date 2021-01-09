class ProgressTracker:
    def __init__(self):
        """
        Keeps track of global number of iterations and number of iterations withing an epoch.
        Epoch is defined by the current sequence.
        Use as a wrapper around an iterator or a sequence.
        >>> tracker = ProgressTracker()
        >>> for i in tracker.track((1,2,3)):
        ...     assert tracker.cnt_total_iter == i
        ...     assert tracker.cnt_iter_in_epoch == i
        >>> # counts are preserved outside of the loop
        >>> assert tracker.cnt_total_iter == 3
        >>> assert tracker.cnt_iter_in_epoch == 3
        >>> # counting continues for the next sequence.
        >>> for i in tracker.track((1,2,3,4)):
        ...     assert tracker.cnt_total_iter == i + 3
        ...     assert tracker.cnt_iter_in_epoch == i
        >>> assert tracker.cnt_total_iter == 7
        >>> assert tracker.cnt_iter_in_epoch == 4
        """
        self.cnt_total_iter = 0
        self.cnt_iter_in_epoch = 0
        self._current_iterable = None

    def track(self, dl):
        self._current_iterable = dl
        return self

    def __next__(self):
        if self._current_iterable is None:
            raise ValueError("Iteration was not initialized. Use track(iterable) to start iteration.")
        result = next(self._current_iterator)
        self.cnt_total_iter += 1
        self.cnt_iter_in_epoch += 1
        return result

    def __iter__(self):
        self.cnt_iter_in_epoch = 0
        self._current_iterator = iter(self._current_iterable)
        return self

    def __len__(self):
        if self._current_iterable is None:
            raise ValueError("Iteration was not initialized. Use track(iterable) to start iteration.")
        return len(self._current_iterable)



