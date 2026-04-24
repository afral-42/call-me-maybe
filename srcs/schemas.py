from pydantic import BaseModel
from enum import Enum


class DataType(Enum):
    BOOL = "boolean"
    NUMBER = "number"
    STRING = "string"
    INTEGER = "integer"


class ParameterDetail(BaseModel):
    type: DataType


class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDetail]
    returns: ParameterDetail


class FunctionCallSchema(BaseModel):
    name: str
    parameters: dict


class PromptSchema(BaseModel):
    prompt: str
