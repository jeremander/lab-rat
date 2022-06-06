from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from fancy_dataclass import SQLDataclass
from typing import Generic, Iterator, TypeVar

from labrat.experiment import Experiment, Result

S = TypeVar('S')  # state
I = TypeVar('I')  # input
R = TypeVar('R', bound = Result)  # result
T = TypeVar('T')


@dataclass  # type: ignore
class StateMachine(ABC, SQLDataclass, Generic[S, I]):
    """Abstract class representing a state machine."""
    @abstractproperty
    def start_state(self) -> S:
        """Gets the start state of the machine."""
    @abstractmethod
    def is_final(self, state: S) -> bool:
        """Returns True if the given state is a final state."""
    @abstractmethod
    def transition(self, state: S, input: I) -> S:
        """Transitions from one state to another, given input."""

@dataclass  # type: ignore
class StateMachineExperiment(Experiment[R], Generic[S, I, R]):
    state_machine: StateMachine[S, I]
    @abstractmethod
    def generate_inputs(self) -> Iterator[I]:
        """Generates a sequence of inputs to the state machine."""
    @abstractmethod
    def get_result(self, state: S) -> R:
        """Gets the result for the given state."""
    def run(self) -> R:
        state = self.state_machine.start_state
        for val in self.generate_inputs():
            if self.state_machine.is_final(state):
                break
            state = self.state_machine.transition(state, val)
        # finish looping if a final state is reached, or input sequence is exhausted
        return self.get_result(state)
