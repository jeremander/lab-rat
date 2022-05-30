from dataclasses import Field, fields, MISSING
from sqlalchemy import Boolean, Column, Integer, Numeric, String, Table
from sqlalchemy.orm import registry
from typing import Any, Callable, ClassVar, Container, Dict, List, Optional, Type, TypeVar, Union

from labrat import JSONDict

T = TypeVar('T')
ColumnMap = Dict[str, Column]

mapper_registry = registry()

def safe_update(d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
    """Updates the first dict with the second.
    Raises a ValueError if any keys overlap."""
    for (key, val) in d2.items():
        if (key in d1):
            raise ValueError(f'duplicate key {key!r}')
        d1[key] = val

class DictDataclass:
    """Mixin for providing conversion between dataclasses and dicts."""
    # if this is True, DictDaclass subfields will have nested dicts
    nested_dict: ClassVar[bool] = True
    def to_dict(self) -> JSONDict:
        d: JSONDict = {}
        for field in fields(self):
            val = getattr(self, field.name)
            if isinstance(val, DictDataclass):  # nested DictDataclass
                val = val.to_dict()
                if (not self.nested_dict):  # merge instead of nesting
                    safe_update(d, val)
                    continue
            d[field.name] = val
        return d
    @classmethod
    def from_dict(cls: Type[T], d: JSONDict) -> T:
        if (not isinstance(d, dict)):
            raise TypeError(f'{cls.__name__}.from_dict argument should be a dict, but it has type {type(d).__name__}')
        args = []
        for field in fields(cls):
            is_mandatory_field = (field.default is MISSING) and (field.default_factory is MISSING)
            if is_mandatory_field or (field.name in d):
                val = d.get(field.name)
            else:  # get default value
                if (field.default is MISSING):  # use default_factory
                    val = field.default_factory()
                else:  # use default
                    val = field.default
            tp = field.type
            origin = getattr(tp, '__origin__', None)
            if origin:
                if (origin is Union):
                    tp = tp.__args__[0]
                elif issubclass(origin, Container):
                    raise NotImplementedError('nested containers not supported')
                else:  # a generic type with parameters
                    tp = origin
            if issubclass(tp, DictDataclass):
                if cls.nested_dict:
                    val = tp.from_dict(val)
                else:  # extract from outer dict instead of nested dict
                    val = tp.from_dict(d)
            if is_mandatory_field and (origin is not Optional) and (val is None):
                # if field is required but not provided, raise an error
                raise TypeError(f'{field.name!r} is a mandatory field, but value was not provided')
            args.append(val)
        return cls(*args)
    @classmethod
    def get_fields(cls) -> List[Field]:
        flds = []
        for field in fields(cls):
            try:
                is_nested = issubclass(field.type, DictDataclass) and field.type.nested_dict
            except TypeError:
                is_nested = False
            if is_nested:  # expand nested subfields
                flds += field.type.get_fields()
            else:
                flds.append(field)
        return flds

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
            if (not field.metadata.get('orm', True)):
                # skip fields whose metadata's 'orm' field is False
                continue
            tp = field.type
            origin = getattr(tp, '__origin__', None)
            if origin:  # compound type
                if (origin is Union):  # use the first type of a Union (also handles Optional)
                    tp = tp.__args__[0]
                elif issubclass(origin, Container):
                    raise TypeError('cannot create ORM table for container type')
                else:  # a generic type with parameters
                    tp = origin
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
