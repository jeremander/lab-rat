from dataclasses import dataclass
from functools import reduce
import operator
import itertools
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Union

from labrat import JSONDict


@dataclass
class ParamGrid(Iterable[JSONDict]):
    """Class representing a collection of experimental parameters, each one given by a JSON dict.
    This stores a dict mapping from keys to lists of values; this represents the Cartesian product of parameters for each key."""
    grid: Dict[str, List[Any]]
    def __init__(self, grid: Dict[str, List[Any]]) -> None:
        # validate
        assert isinstance(grid, dict), 'ParamGrid entry must be a dict'
        for (key, val) in grid.items():
            assert isinstance(key, str), 'ParamGrid keys must be strings'
            assert isinstance(val, Sequence), 'ParamGrid values must be lists'
        self.grid = grid
    def __iter__(self) -> Iterator[JSONDict]:
        keys = list(self.grid)
        for vals in itertools.product(*self.grid.values()):
            yield dict(zip(keys, vals))
    @classmethod
    def constant(cls, params: JSONDict) -> 'ParamGrid':
        """Constructor from a single parameter dict (i.e. a trivial Cartesian product)."""
        return cls({key : [val] for (key, val) in params.items()})
    @classmethod
    def product(cls, *grids: 'ParamGrid') -> 'ParamGrid':
        """Constructs the Cartesian product of multiple ParamGrids, with the resulting dicts merged together.
        Raises a ValueError if any keys overlap."""
        assert (len(grids) >= 1), 'must have at least one ParamGrid'
        keys = set(grids[0].grid)
        for grid in grids[1:]:
            intersection = list(keys.intersection(grid.grid))
            if intersection:
                raise ValueError(f'cannot make Cartesian product of ParamGrids with overlapping key {intersection[0]!r}')
            keys.update(grid.grid)
        return cls(reduce(operator.or_, (grid.grid for grid in grids)))
    def __mul__(self, other: 'ParamGrid') -> 'ParamGrid':
        """Returns the Cartesian product of two ParamGrids, with the resulting dicts merged together.
        Raises a ValueError if keys overlap."""
        return self.__class__.product(self, other)

@dataclass
class Params(Iterable[JSONDict]):
    """Class representing a collection of experimental parameters, each one given by a JSON dict.
    This stores a list of ParamGrid objects; this represents the union of Cartesian products of parameters."""
    grids: List[ParamGrid]
    def __init__(self, grids: Union['Params', ParamGrid, List[ParamGrid], JSONDict, List[JSONDict]]) -> None:
        if isinstance(grids, Params):
            self.grids = grids.grids
        else:
            grid_list = grids if isinstance(grids, list) else [grids]
            # wrap in ParamGrid, if not already
            self.grids = [grid if isinstance(grid, ParamGrid) else ParamGrid(grid) for grid in grid_list]
    def __iter__(self) -> Iterator[JSONDict]:
        return itertools.chain.from_iterable(iter(grid) for grid in self.grids)
    def __mul__(self, other: 'Params') -> 'Params':
        return self.__class__([grid1 * grid2 for (grid1, grid2) in itertools.product(self.grids, other.grids)])
    def __add__(self, other: 'Params') -> 'Params':
        return self.__class__(self.grids + other.grids)
