from srcs.states import (
    DynamicStringState,
    NumberState,
    State,
    StaticStringState,
    StringRouterState
)
from srcs.schemas import FunctionSchema, PromptSchema
from srcs.schemas import DataType
from srcs.prompts import Prompt
import json
from srcs.machine import StateMachine
from pydantic import ValidationError


class ParsingException(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Parsing error: {detail}")


def parse_functions(functions: list[dict]) -> list[FunctionSchema]:
    try:
        function_schemas: list[FunctionSchema] = [
            FunctionSchema.model_validate(function) for function in functions
        ]
    except ValidationError as e:
        raise ParsingException(str(e))

    return function_schemas


def get_boolean_automate() -> StringRouterState:
    true_state = StaticStringState("true")
    false_state = StaticStringState("false")

    return StringRouterState(
        {true_state, false_state}, {true_state: [], false_state: []}
    )


def build_function_states(functions: list[dict]) -> StringRouterState:
    function_schemas = parse_functions(functions)

    routers: list[StaticStringState] = []
    next_states: dict[StaticStringState, list[State]] = {}

    for schema in function_schemas:
        router = StaticStringState(schema.name)
        local_next_states: list[State] = [
            StaticStringState("\",\"parameters\":{")
        ]

        param_numbers = len(schema.parameters)
        if not param_numbers:
            local_next_states.append(StaticStringState("}"))
            continue

        last = param_numbers - 1

        for i, (name, type) in enumerate(schema.parameters.items()):
            local_next_states.append(StaticStringState(
                f"\"{name}\":"
            ))
            if type.type == DataType.NUMBER:
                local_next_states.append(
                    NumberState("}" if i == param_numbers - 1 else ",")
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


def build_prompts(prompts_path: str, functions: str) -> list[Prompt]:
    prompts: list[Prompt] = []

    try:
        with open(prompts_path) as f:
            texts = json.load(f)
    except json.JSONDecodeError:
        raise ParsingException(
            "Invalid JSON format for prompts file"
        )
    except PermissionError:
        raise ParsingException(
            "Invalid permissions for reading prompts file"
        )
    except FileNotFoundError:
        raise ParsingException(
            "Prompts file doesn't exist"
        )
    except Exception:
        raise ParsingException(
            "Unexpected error reading prompts file"
        )

    try:
        for text in texts:
            PromptSchema.model_validate(text)
            prompts.append(Prompt(text, functions))

        return prompts
    except ValidationError:
        raise ParsingException("Invalid prompt format")


def build_state_machine(functions: list[dict]) -> StateMachine:
    begin = StaticStringState("{\"name\":\"")
    router = build_function_states(functions)

    return StateMachine([begin, router])


def build_linked_state_machines(
    functions_definition_path: str,
    input_file_path: str,
) -> dict[Prompt, StateMachine]:
    linked_state_machines: dict[Prompt, StateMachine] = {}

    try:
        with open(functions_definition_path) as f:
            funcs = json.load(f)

    except json.JSONDecodeError:
        raise ParsingException(
            "Invalid JSON format for prompts file"
        )
    except PermissionError:
        raise ParsingException(
            "Invalid permissions for reading prompts file"
        )
    except FileNotFoundError:
        raise ParsingException(
            "Prompts file doesn't exist"
        )
    except Exception as e:
        raise ParsingException(
            f"Unexpected error reading prompts file {e}"
        )

    prompts = build_prompts(input_file_path, funcs)
    for prompt in prompts:
        linked_state_machines[prompt] = build_state_machine(funcs)

    return linked_state_machines
