from abc import ABC, abstractclassmethod, abstractmethod
import bdb
from dataclasses import dataclass, fields, make_dataclass
from datetime import datetime
from functools import cache, partial
from logging import Logger
import multiprocessing as mp
from sqlalchemy import Column, Integer, String
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from typing import Generic, Iterator, Optional, Type, TypeVar

from labrat import get_logger, JSONDict
from labrat.orm import ColumnMap, mapper_registry, ORMDataclass, orm_table
from labrat.params import Params


T = TypeVar('T')

ID_COLUMN = Column('id', Integer, primary_key = True, autoincrement = True)
TIME_COLUMN = Column('time', String)


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
    def extra_orm_columns(cls) -> ColumnMap:
        """Gets additional columns to provide to the ORM table which are not included among the dataclass fields."""
        return {'id' : ID_COLUMN, 'time' : TIME_COLUMN}
    @classmethod
    @cache
    def orm_cls(cls) -> Type[ORMDataclass]:
        """Creates a custom subclass of ORMDataclass that has a sqlalchemy-backed SQL table."""
        flds = {}
        for cl in [cls, cls.result_cls()]:  # type: ignore
            for field in fields(cl):
                if (field.name in flds):
                    raise ValueError(f'duplicate field name {field.name!r}')
                flds[field.name] = field.type
        dcl = make_dataclass('ExperimentWithResult', list(flds.items()), bases = (ORMDataclass,))
        dcl.__name__ = cls.__name__
        cols = cls.extra_orm_columns()
        return orm_table(cols)(dcl)
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
        cls = self.experiment_cls.orm_cls()
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

