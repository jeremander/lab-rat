from abc import abstractclassmethod, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
import itertools
import multiprocessing as mp
from sqlalchemy import Boolean, Column, create_engine, Integer, Numeric, String, Table
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import registry, sessionmaker
from tqdm import tqdm
from typing import Any, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union

# cartesian product of parameter values (specified by keys and value lists)
ParamGrid = Dict[str, List[Any]]
# a ParamGrid, or a union of them
Params = Union[ParamGrid, List[ParamGrid]]
JSONDict = Dict[str, Any]

mapper_registry = registry()

def get_column_type(tp: type) -> type:
    if issubclass(tp, str):
        return String
    elif issubclass(tp, bool):
        return Boolean
    elif issubclass(tp, int):
        return Integer
    elif issubclass(tp, float):
        return Numeric
    raise TypeError(f'could not convert type {tp!r} to SQL column type')

class ORMDataclass:
    @classmethod
    def get_columns(cls) -> Dict[str, Column]:
        cols = {}
        for (key, field) in cls.__dataclass_fields__.items():
            if issubclass(field.type, ORMDataclass):  # nested ORMDataclass
                cols.update(field.type.get_columns())
            else:
                cols[key] = Column(key, get_column_type(field.type))
        return cols
    def to_dict(self) -> JSONDict:
        d = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            if isinstance(val, ORMDataclass):
                val = val.to_dict()
            d[key] = val
        return d
    @classmethod
    def from_dict(cls, d: JSONDict) -> 'ORMDataclass':
        # assume the values are in order (TODO: improve this)
        args = []
        for (key, field) in cls.__dataclass_fields__.items():
            val = d.get(key)
            if issubclass(field.type, ORMDataclass):
                val = field.type.from_dict(val)
            args.append(val)
        return cls(*args)

class Result(ORMDataclass):
    """A class of experimental result that can be mapped to a SQL table."""

R = TypeVar('R', bound = Result)

class ExperimentWithResult(ORMDataclass):
    """Base class storing an experiment with a result."""
    experiment: ORMDataclass
    result: ORMDataclass

class Experiment(ORMDataclass, Generic[R]):
    @abstractclassmethod
    def result_cls(cls) -> Type[R]:
        """Gets the Result subclass."""
    @classmethod
    @lru_cache
    def experiment_with_result_cls(cls) -> type:
        """Creates a custom subclass of ExperimentWithResult that has a sqlalchemy-backed SQL table."""
        @dataclass
        class _ExperimentWithResult(ExperimentWithResult):
            pass
        fields = {
            'id' : Column('id', Integer, primary_key = True, autoincrement = True),
            'time' : Column('time', String),
        }
        for cls_ in [cls, cls.result_cls()]:
            for (key, col) in cls_.get_columns().items():
                if (key in fields):
                    raise ValueError(f'duplicate field name {key!r}')
                fields[key] = col
        _ExperimentWithResult.__table__ = Table(cls.__name__.lower(), mapper_registry.metadata, *fields.values())
        return mapper_registry.mapped(_ExperimentWithResult)
    @abstractmethod
    def _run(self) -> R:
        """Runs the experiment, producing a Result."""
    def run(self, errors: str = 'warn') -> Optional[JSONDict]:
        try:
            result = self._run()
            result.time = datetime.now().isoformat()
            return {**self.to_dict(), **result.to_dict()}
        except Exception as e:
            if (errors == 'raise'):
                raise e
            if (errors == 'warn'):
                print(f'WARNING: {e.__class__.__name__}: {e}')
                return None

@dataclass
class ExperimentRunner:
    experiment_cls: Type[Experiment]
    engine: Engine  # SQL engine
    params: Params
    verbosity: int = 0  # verbosity level
    errors: str = 'warn'  # how to handle errors (ignore, warn, raise)
    num_threads: int = 1  # number of threads to use
    chunk_size: int = 1  # number of experiments per chunk
    def __iter__(self) -> Iterator[Experiment]:
        grids = self.params if isinstance(self.params, list) else [self.params]
        for grid in grids:
            for vals in itertools.product(*grid.values()):
                yield self.experiment_cls(*vals)
    def run(self) -> None:
        cls = self.experiment_cls.experiment_with_result_cls()
        mapper_registry.metadata.create_all(self.engine)
        Session = sessionmaker(bind = self.engine)
        session = Session()
        experiments = list(self)
        num_experiments = len(experiments)
        print(f'Running {num_experiments} experiments with {self.num_threads} thread(s)...')
        pool = mp.Pool(self.num_threads)
        mapper = map if (self.num_threads == 1) else partial(pool.imap_unordered, chunksize = self.chunk_size)
        func = partial(Experiment.run, errors = self.errors)
        results = mapper(func, experiments)
        for result in tqdm(results, total = num_experiments):
            if (result is not None):
                session.add(cls.from_dict(result))
                session.commit()


#######

@dataclass
class MyResult(Result):
    d: int
    e: bool

@dataclass
class MyExperiment(Experiment[MyResult]):
    a: int
    b: str
    c: float
    @classmethod
    def result_cls(cls) -> Type[MyResult]:
        return MyResult
    def _run(self) -> MyResult:
        import time
        time.sleep(0.01)
        if (self.a == 33):
            raise ValueError('33 is bad!')
        return MyResult(self.a, self.b == 'b')

if __name__ == '__main__':
    engine = create_engine('sqlite:///test.sqlite')
    params = {'a' : list(range(100)), 'b' : ['a', 'b', 'c', 'd'], 'c': [1.0, 10.0]}
    runner = ExperimentRunner(MyExperiment, engine, params, num_threads = 4)
    runner.run()
