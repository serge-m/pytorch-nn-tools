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

    def track(self, dl):
        def tracked_iterator():
            self.cnt_iter_in_epoch = 0
            for x in dl:
                self.cnt_total_iter += 1
                self.cnt_iter_in_epoch += 1
                yield x

        return tracked_iterator()
