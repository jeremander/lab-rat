from abc import ABC, abstractclassmethod, abstractmethod
import bdb
from dataclasses import dataclass, make_dataclass
from datetime import datetime
from functools import cache, partial
import itertools
from logging import Logger
import multiprocessing as mp
from sqlalchemy import Boolean, Column, Integer, Numeric, String, Table
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import registry, sessionmaker
from tqdm import tqdm
from typing import Dict, Generic, Iterator, Optional, Type, TypeVar

from labrat import get_logger, JSONDict
from labrat.params import Params


T = TypeVar('T')

mapper_registry = registry()

def get_column_type(tp: type) -> type:
    """Given a Python type, returns a corresponding sqlalchemy column type."""
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
        for (key, field) in cls.__dataclass_fields__.items():  # type: ignore
            tp = field.type
            if hasattr(tp, '__origin__'):
                tp = tp.__args__[0]  # get the first type of a Union (or Optional)
            if issubclass(tp, ORMDataclass):  # nested ORMDataclass
                cols.update(tp.get_columns())
            else:
                cols[key] = Column(key, get_column_type(tp))
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
    def from_dict(cls: Type[T], d: JSONDict) -> T:
        args = []
        for (key, field) in cls.__dataclass_fields__.items():
            val = d.get(key)
            tp = field.type
            if hasattr(tp, '__origin__'):
                tp = tp.__args__[0]
            if issubclass(tp, ORMDataclass):
                val = tp.from_dict(val)
            args.append(val)
        return cls(*args)

def orm_table(cls: Type[ORMDataclass]) -> Type[ORMDataclass]:
    fields = {
        'id' : Column('id', Integer, primary_key = True, autoincrement = True),
        'time' : Column('time', String),
    }
    for (key, col) in cls.get_columns().items():
        if (key in fields):
            raise ValueError(f'duplicate field name {key!r}')
        fields[key] = col
    cls.__table__ = Table(cls.__name__.lower(), mapper_registry.metadata, *fields.values())
    return cls

class Result(ORMDataclass):
    """A class of experimental result that can be mapped to a SQL table."""

R = TypeVar('R', bound = Result)

class Experiment(ORMDataclass, Generic[R], ABC):
    @classmethod
    @property
    def logger(cls) -> Logger:
        return get_logger(cls.__name__)
    @abstractclassmethod
    def result_cls(cls) -> Type[R]:
        """Gets the Result subclass."""
    @classmethod
    @cache
    def experiment_with_result_cls(cls) -> type:
        """Creates a custom subclass of ORMDataclass that has a sqlalchemy-backed SQL table."""
        fields = {}
        for cl in [cls, cls.result_cls()]:  # type: ignore
            for (key, field) in cl.__dataclass_fields__.items():
                if (key in fields):
                    raise ValueError(f'duplicate field name {key!r}')
                fields[key] = field.type
        dcl = make_dataclass('ExperimentWithResult', list(fields.items()), bases = (ORMDataclass,))
        dcl.__name__ = cls.__name__
        dcl = orm_table(dcl)
        return mapper_registry.mapped(dcl)
    @abstractmethod
    def run(self) -> R:
        """Runs the experiment, producing a Result."""

def run_experiment(experiment: Experiment, errors: str = 'raise') -> Optional[JSONDict]:  # type: ignore
    try:
        result = experiment.run()
        return {'time' : datetime.now().isoformat(), **experiment.to_dict(), **result.to_dict()}
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, bdb.BdbQuit)):
            raise e
        if (errors == 'raise'):
            raise e
        if (errors == 'warn'):
            experiment.logger.error(f'{e.__class__.__name__}: {e}')
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
    def __post_init__(self) -> None:
        # ensure params are wrapped in the Params class
        self.params = Params(self.params)
    def __iter__(self) -> Iterator[Experiment]:
        yield from (self.experiment_cls.from_dict(d) for d in self.params)
    def run(self) -> None:
        cls = self.experiment_cls.experiment_with_result_cls()
        mapper_registry.metadata.create_all(self.engine)
        Session = sessionmaker(bind = self.engine)
        session = Session()
        experiments = list(self)
        num_experiments = len(experiments)
        logger = self.experiment_cls.logger
        logger.info(f'Running {num_experiments} experiments with {self.num_threads} thread(s)...')
        pool = mp.Pool(self.num_threads)
        mapper = map if (self.num_threads == 1) else partial(pool.imap_unordered, chunksize = self.chunk_size)
        func = partial(run_experiment, errors = self.errors)
        results = mapper(func, experiments)  # type: ignore
        for result in tqdm(results, total = num_experiments):
            if (result is not None):
                time = result.pop('time')
                res = cls.from_dict(result)
                res.time = time
                session.add(res)
                session.commit()
        logger.info('DONE!')

