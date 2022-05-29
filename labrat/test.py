from dataclasses import dataclass
from sqlalchemy import create_engine
from typing import Type

from labrat import Experiment, ExperimentRunner, Result


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
    def run(self) -> MyResult:
        # import time
        # time.sleep(0.01)
        if (self.a == 33):
            raise ValueError('33 is bad!')
        return MyResult(self.a, self.b == 'b')


if __name__ == '__main__':

    engine = create_engine('sqlite:///test.sqlite')
    params = {'a' : list(range(100)), 'b' : ['a', 'b', 'c', 'd'], 'c': [1.0, 10.0]}
    runner = ExperimentRunner(MyExperiment, engine, params, num_threads = 1)
    runner.run()