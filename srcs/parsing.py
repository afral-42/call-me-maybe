import json
from pydantic import BaseModel


class ParsingException(Exception):
    pass


@dataclass
class FunctionMetadata:
    names: str
    parameters: list[parameters]


def parse_functions(functions_path: str) -> None:
    with open(functions_path) as f:
        functions = json.load(f)

    names = []

    try:
        for function in functions:
            names.append(function["name"])

    except Exception as e:
        raise ParsingException(str(e))
