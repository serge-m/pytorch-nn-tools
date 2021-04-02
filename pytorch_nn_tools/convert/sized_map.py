from collections.abc import Iterable, Sequence, Sized, Collection
from typing import Union, Callable


def sized_map(fn: Callable, iterable: Union[Iterable, Collection]) -> Union[Iterable, Collection]:
    """
    Like a standard `map`, but maintains the size.
    It's a convenience function to use with tqdm for example.

    >>> list(sized_map(str.upper, ('a','b','c')))
    ['A', 'B', 'C']
    >>> len(sized_map(lambda x: x**2, [8,2])) # must have length if input has length
    2
    >>> len(sized_map(lambda x: x**2, range(3))) # must have length if input has length
    3
    >>> list(sized_map(lambda x: x**2, (i for i in range(3)))) # also works with non-sized
    [0, 1, 4]
    >>> len(sized_map(lambda x: x**2, (i for i in range(3)))) # len fails if the iterable is not sized
    Traceback (most recent call last):
    ...
    TypeError: object of type 'generator' has no len()
    """
    return MappedSeq(fn, iterable)


class MappedSeq(Iterable, Sized):

    def __init__(self, fn: Callable, iterable: Union[Iterable, Sequence]):
        """
        Implements map() maintaining length property.
        >>> list(MappedSeq(str.upper, ('a','b','c')))
        ['A', 'B', 'C']
        >>> len(MappedSeq(lambda x: x**2, [8,2])) # must have length if input has length
        2
        >>> len(MappedSeq(lambda x: x**2, range(3))) # must have length if input has length
        3
        >>> list(MappedSeq(lambda x: x**2, (i for i in range(3)))) # also works with non-sized
        [0, 1, 4]
        >>> len(MappedSeq(lambda x: x**2, (i for i in range(3)))) # len fails if the iterable is not sized
        Traceback (most recent call last):
        ...
        TypeError: object of type 'generator' has no len()
        """
        self._iterable = iterable
        self._fn = fn

    def __iter__(self):
        return map(self._fn, self._iterable)

    def __len__(self):
        return len(self._iterable)
