from abc import ABC, abstractclassmethod, abstractmethod
import bdb
from copy import copy
from dataclasses import dataclass, make_dataclass, MISSING
from datetime import datetime
from fancy_dataclass.sql import ColumnMap, DEFAULT_REGISTRY, register, SQLDataclass
from functools import cache, partial
from logging import Logger
import multiprocessing as mp
import random
from sqlalchemy import Column, Integer, String
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm
from typing import Dict, Generic, Iterator, Optional, Tuple, Type, TypeVar

from labrat import get_logger, JSONDict, LOGGER
from labrat.params import Params


T = TypeVar('T')

class Result(SQLDataclass):
    """A class of experimental result that can be mapped to a SQL table."""

R = TypeVar('R', bound = Result)

class Experiment(SQLDataclass, Generic[R], ABC):
    @classmethod
    @property
    def logger(cls) -> Logger:
        return get_logger(cls.__name__)
    @abstractclassmethod
    def result_cls(cls) -> Type[R]:
        """Gets the Result subclass."""
    @classmethod
    def extra_columns(cls) -> ColumnMap:
        """Gets additional columns to provide to the SQL table which are not included among the dataclass fields."""
        return {
            'id' : Column('id', Integer, primary_key = True, autoincrement = True),
            'exp_id' : Column('exp_id', String),
            'time' : Column('time', String)
        }
    @classmethod
    @cache
    def sql_cls(cls) -> Type[SQLDataclass]:
        """Creates a custom subclass of SQLDataclass that has a sqlalchemy-backed SQL table."""
        flds = []
        for cl in [cls, cls.result_cls()]:
            for field in cl.get_fields():
                has_default = (field.default is not MISSING) or (field.default_factory is not MISSING)
                # to preserve order, put a dummy default of None for any mandatory fields
                if (not has_default):
                    field = copy(field)
                    field.default = None
                flds.append((field.name, field.type, field))
        dcl = make_dataclass(cls.__name__, flds, bases = (SQLDataclass,))
        return register(extra_cols = cls.extra_columns())(dcl)
    @abstractmethod
    def run(self) -> R:
        """Runs the experiment, producing a Result."""

def run_experiment(experiment: Experiment, errors: str = 'raise') -> Optional[Tuple[Type[Experiment], JSONDict]]:  # type: ignore
    try:
        result = experiment.run()
        return (experiment.__class__, {'time' : datetime.now().isoformat(), **experiment.to_dict(), **result.to_dict()})
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, bdb.BdbQuit)):
            raise e
        elif (errors == 'raise'):
            raise e
        elif (errors == 'warn'):
            experiment.logger.error(f'{e.__class__.__name__}: {e}')
            return None

@dataclass
class ExperimentRunner:
    """Main driver for running experiments."""
    params: Dict[Type[Experiment], Params]  # mapping from experiment class to parameters
    engine: Engine  # SQL engine
    verbosity: int = 0  # verbosity level
    errors: str = 'warn'  # how to handle errors (ignore, warn, raise)
    num_threads: int = 1  # number of threads to use
    chunk_size: int = 1  # number of experiments per chunk
    shuffle: bool = False  # shuffle the experiments
    def __post_init__(self) -> None:
        # ensure params are wrapped in the Params class
        self.params = {cls : Params(params) for (cls, params) in self.params.items()}
    def __iter__(self) -> Iterator[Experiment]:
        for (cls, params) in self.params.items():
            yield from (cls.from_dict(d) for d in params)
    def result_classes(self) -> Dict[Type[Experiment], type]:
        """Gets a mapping from Experiment classes to SQLDataclasses storing both parameters and results."""
        return {cls : cls.sql_cls() for cls in self.params}
    def create_tables(self) -> None:
        # create all the tables
        for cls in self.params:
            cls.sql_cls()
        DEFAULT_REGISTRY.metadata.create_all(self.engine)
    def make_session(self) -> Session:
        return sessionmaker(bind = self.engine)()
    def run(self) -> None:
        self.create_tables()
        session = self.make_session()
        experiments = list(self)
        if self.shuffle:
            random.shuffle(experiments)
        num_experiments = len(experiments)
        LOGGER.info(f'Running {num_experiments} experiments with {self.num_threads} thread(s)...')
        pool = mp.Pool(self.num_threads)
        mapper = map if (self.num_threads == 1) else partial(pool.imap_unordered, chunksize = self.chunk_size)
        func = partial(run_experiment, errors = self.errors)
        results = mapper(func, experiments)  # type: ignore
        exp_id = datetime.now().strftime('%Y%m%d%H%M%S')
        result_classes = self.result_classes()
        for result in tqdm(results, total = num_experiments):
            if (result is not None):
                (cls, d) = result
                time = d.pop('time')
                res = result_classes[cls].from_dict(d)  # automatically infers subtype
                res.exp_id = exp_id
                res.time = time
                session.add(res)
                session.commit()
        LOGGER.info('\033[1mDONE!')

