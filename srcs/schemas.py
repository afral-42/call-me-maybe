from pydantic import BaseModel
from enum import Enum


class DataType(Enum):
    BOOL = "boolean"
    NUMBER = "number"
    STRING = "string"


class ParameterDetail(BaseModel):
    type: DataType


class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDetail]
