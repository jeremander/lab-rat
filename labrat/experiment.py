from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import itertools
import multiprocessing as mp
from sqlalchemy import Boolean, Column, create_engine, Integer, Numeric, String, Table
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import registry, sessionmaker
from tqdm import tqdm
from typing import Any, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union

R = TypeVar('R')
# cartesian product of parameter values (specified by keys and value lists)
ParamGrid = Dict[str, List[Any]]
# a ParamGrid, or a union of them
Params = Union[ParamGrid, List[ParamGrid]]

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

def orm_dataclass(cls: type) -> type:
    dcls: type = dataclass(cls)
    name = cls.__name__.lower()
    fields = [
        Column('id', Integer, primary_key = True, autoincrement = True),
        Column('time', String)
    ]
    for (key, field) in dcls.__dataclass_fields__.items():
        fields.append(Column(key, get_column_type(field.type)))
    dcls.__table__ = Table(name, mapper_registry.metadata, *fields)
    return mapper_registry.mapped(dcls)

class Result:
    """A class of experimental result that can be mapped to a SQL table."""

class Experiment(Generic[R]):
    @abstractmethod
    def run(self) -> R:
        """Runs the experiment, producing a Result."""
    def run_and_timestamp(self, errors: str = 'warn') -> Optional[R]:
        try:
            result = self.run()
            result.time = datetime.now().isoformat()
            return result
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
        mapper_registry.metadata.create_all(self.engine)
        Session = sessionmaker(bind = self.engine)
        session = Session()
        experiments = list(self)
        num_experiments = len(experiments)
        print(f'Running {num_experiments} experiments with {self.num_threads} thread(s)...')
        pool = mp.Pool(self.num_threads)
        mapper = map if (self.num_threads == 1) else partial(pool.imap_unordered, chunksize = self.chunk_size)
        func = partial(Experiment.run_and_timestamp, errors = self.errors)
        results = mapper(func, experiments)
        for result in tqdm(results, total = num_experiments):
            if (result is not None):
                session.add(result)
                session.commit()


#######

@orm_dataclass
class MyResult(Result):
    a: int
    b: str
    c: float

@dataclass
class MyExperiment(Experiment[MyResult]):
    a: int
    b: str
    c: float
    def run(self) -> MyResult:
        import time
        time.sleep(0.01)
        if (self.a == 33):
            raise ValueError('33 is bad!')
        return MyResult(self.a, self.b, self.c)

if __name__ == '__main__':
    engine = create_engine('sqlite:///test.sqlite')
    params = {'a' : list(range(100)), 'b' : ['a', 'b', 'c', 'd'], 'c': [1.0, 10.0]}
    runner = ExperimentRunner(MyExperiment, engine, params, num_threads = 4)
    runner.run()
