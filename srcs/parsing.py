from srcs.states import DynamicStringState, NumberState, State, StaticStringState, StringRouterState, get_boolean_automate
from srcs.schemas import FunctionSchema
from srcs.schemas import DataType
import json
from pydantic import ValidationError


class ParsingException(Exception):
    pass


def parse_functions(functions_path: str) -> list[FunctionSchema]:
    with open(functions_path) as f:
        functions = json.load(f)

    try:
        function_schemas: list[FunctionSchema] = [
            FunctionSchema.model_validate(function) for function in functions
        ]
    except ValidationError as e:
        raise ParsingException(str(e))

    return function_schemas


def build_function_states(functions_path: str) -> StringRouterState:
    function_schemas = parse_functions(functions_path)

    routers: list[StaticStringState] = []
    next_states: dict[StaticStringState, list[State]] = {}

    for schema in function_schemas:
        router = StaticStringState(schema.name)
        local_next_states: list[State] = [
            StaticStringState("\",\"parameters\":{")
        ]

        last = len(schema.parameters) - 1

        for i, (name, type) in enumerate(schema.parameters.items()):
            local_next_states.append(StaticStringState(
                f"\"{name}\":"
            ))
            if type.type == DataType.NUMBER:
                local_next_states.append(
                    NumberState("}" if i == last else ",")
                )
                if i == last:
                    local_next_states.append(StaticStringState("}"))
            elif type.type == DataType.STRING:
                local_next_states.append(StaticStringState("\""))
                local_next_states.append(DynamicStringState())

                if i != last:
                    local_next_states.append(StaticStringState(","))
                else:
                    local_next_states.append(StaticStringState("}}"))
            elif type.type == DataType.BOOL:
                local_next_states.append(get_boolean_automate())
                if i != last:
                    local_next_states.append(StaticStringState(","))
                else:
                    local_next_states.append(StaticStringState("}}"))

        routers.append(router)
        next_states[router] = local_next_states

    return StringRouterState(set(routers), next_states)
