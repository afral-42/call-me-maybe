from pydantic import BaseModel
from enum import Enum


class DataType(Enum):
    """

    Enumeration of supported data types for function parameters.

    """

    BOOL = "boolean"
    NUMBER = "number"
    STRING = "string"
    INTEGER = "integer"


class ParameterDetail(BaseModel):
    """

    Detail of a function parameter.

    Attributes:

        type (DataType): The type of the parameter.

    """

    type: DataType


class FunctionSchema(BaseModel):
    """

    Schema for a function definition.

    Attributes:

        name (str): The function name.

        description (str): Description of the function.

        parameters (dict[str, ParameterDetail]): Parameter details.

        returns (ParameterDetail): Return type detail.

    """

    name: str
    description: str
    parameters: dict[str, ParameterDetail]
    returns: ParameterDetail


class PromptSchema(BaseModel):
    """

    Schema for a prompt.

    Attributes:

        prompt (str): The prompt text.

    """

    prompt: str
