from dataclasses import fields
from sqlalchemy import Boolean, Column, Integer, Numeric, String, Table
from sqlalchemy.orm import registry
from typing import Callable, Dict, Type, TypeVar

from labrat import JSONDict

T = TypeVar('T')
ColumnMap = Dict[str, Column]

mapper_registry = registry()


class DictDataclass:
    """Mixin for providing conversion between dataclasses and dicts."""
    def to_dict(self) -> JSONDict:
        d = {}
        for field in fields(self):
            val = getattr(self, field.name)
            if isinstance(val, DictDataclass):  # nested DictDataclass
                val = val.to_dict()
            d[field.name] = val
        return d
    @classmethod
    def from_dict(cls: Type[T], d: JSONDict) -> T:
        args = []
        for field in fields(cls):
            val = d.get(field.name)
            tp = field.type
            if hasattr(tp, '__origin__'):  #
                tp = tp.__args__[0]
            if issubclass(tp, DictDataclass):
                val = tp.from_dict(val)
            args.append(val)
        return cls(*args)

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

class ORMDataclass(DictDataclass):
    @classmethod
    def get_columns(cls) -> ColumnMap:
        cols = {}
        for field in fields(cls):
            tp = field.type
            if hasattr(tp, '__origin__'):
                tp = tp.__args__[0]  # get the first type of a Union (or Optional)
            # TODO: handle compound type
            if issubclass(tp, ORMDataclass):  # nested ORMDataclass
                cols.update(tp.get_columns())
            else:
                cols[field.name] = Column(field.name, get_column_type(tp))
        return cols

def _get_orm_table(cls: Type[ORMDataclass], cols: ColumnMap = {}) -> Table:
    cols = dict(cols)
    for (key, col) in cls.get_columns().items():
        if (key in cols):
            raise ValueError(f'duplicate field name {key!r}')
        cols[key] = col
    return Table(cls.__name__.lower(), mapper_registry.metadata, *cols.values())

def orm_table(cols: ColumnMap = {}) -> Callable[[Type[ORMDataclass]], Type[ORMDataclass]]:
    """Decorator that provides a sqlalchemy table for an ORMDataclass."""
    def _orm_table(cls: Type[ORMDataclass]) -> Type[ORMDataclass]:
        cls.__table__ = _get_orm_table(cls, cols)
        return mapper_registry.mapped(cls)
    return _orm_table
