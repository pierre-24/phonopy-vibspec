"""
Simulate IR and Raman spectra with Phonopy
"""

import logging
import os

__version__ = '0.4.0'
__author__ = 'Pierre Beaujean'
__maintainer__ = 'Pierre Beaujean'
__email__ = 'pierre.beaujean@unamur.be'
__status__ = 'Development'

from typing import Optional, Set

# logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', 'WARNING').upper())


class GetListWithinBounds:
    """
    Get a list of integer within an interval `[min, max]`.
    Open intervals are possible by setting `None`.
    """
    def __init__(self, min_index: Optional[int] = None, max_index: Optional[int] = None):
        self.min_index = min_index
        self.max_index = max_index

    def _check_or_raise(self, n: int):
        """Check that number fall in interval"""
        if self.min_index is not None and n < self.min_index:
            raise ValueError('{} should be larger than {}'.format(n, self.min_index))
        if self.max_index is not None and n > self.max_index:
            raise ValueError('{} should be smaller than {}'.format(n, self.max_index))

    def __call__(self, inp: str) -> Set[int]:
        """
        Get a list of atom indices.
        Ranges (i.e., `1-3` = `[0, 1, 2]`) and final wildcard (i.e., 2-*` = `[2, 3 ...]`, only if closed interval)
        are supported.
        """
        lst = set()

        for x in inp.split():
            if '-' in x:
                chunks = x.split('-')
                if len(chunks) != 2:
                    raise ValueError('{} should contain 2 elements'.format(x))

                b = int(chunks[0])
                self._check_or_raise(b)

                if chunks[1] == '*':
                    if self.max_index is None:
                        raise ValueError('cannot use `*` with an open interval')
                    e = self.max_index
                else:
                    e = int(chunks[1])
                    self._check_or_raise(e)

                lst |= set(range(b, e + 1))
            else:
                n = int(x)
                self._check_or_raise(n)
                lst.add(int(x))

        return lst
