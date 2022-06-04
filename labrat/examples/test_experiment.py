from dataclasses import dataclass
from sqlalchemy import create_engine
import time
from typing import Type

from labrat.experiment import Experiment, ExperimentRunner, Result
from labrat.params import Params


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
        time.sleep(0.1)
        if (self.a == 33):
            raise ValueError('13 is unlucky!')
        return MyResult(self.a, self.b == 'b')


if __name__ == '__main__':

    engine = create_engine('sqlite:///test.sqlite')
    params = Params({'a' : list(range(20)), 'b' : ['a', 'b', 'c'], 'c': [1.0, 10.0]})
    runner = ExperimentRunner(MyExperiment, engine, params, num_threads = 1)
    runner.run()