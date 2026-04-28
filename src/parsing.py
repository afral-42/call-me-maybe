from src.states import (
    DynamicStringState,
    IntegerState,
    NumberState,
    State,
    StaticStringState,
    StringRouterState
)
from src.schemas import FunctionSchema, PromptSchema
from src.schemas import DataType
from src.prompts import Prompt
import json
from src.machine import StateMachine
from pydantic import ValidationError
from typing import Any, cast


class ParsingException(Exception):
    def __init__(self, detail: str) -> None:
        """

        Initialize the parsing exception.

        Args:

            detail (str): The detail message for the error.

        """
        super().__init__(f"Parsing error: {detail}")


def load_json_file(file_path: str, file_type: str) -> list[dict[str, Any]]:
    """

    Load and parse a JSON file.

    Args:

        file_path (str): Path to the JSON file.

        file_type (str): Description of the file type for error messages.

    Returns:

        list[dict[str, Any]]: The parsed JSON data.

    Raises:

        ParsingException: If the file cannot be read or parsed.

    """
    try:
        with open(file_path, "r") as f:
            return cast(list[dict[str, Any]], json.load(f))
    except json.JSONDecodeError:
        raise ParsingException(
            f"Invalid JSON format in {file_type} file: {file_path}"
        )
    except PermissionError:
        raise ParsingException(
            f"Permission denied for reading {file_type} file: {file_path}"
        )
    except FileNotFoundError:
        raise ParsingException(
            f"The {file_type} file does not exist: {file_path}"
        )
    except Exception as e:
        raise ParsingException(
            f"Unexpected error reading {file_type} file ({file_path}): {e}"
        )


def parse_functions(functions: list[dict[str, Any]]) -> list[FunctionSchema]:
    """

    Parse a list of function dictionaries into FunctionSchema objects.

    Args:

        functions (list[dict[str, Any]]): List of function definitions.

    Returns:

        list[FunctionSchema]: List of parsed function schemas.

    Raises:

        ParsingException: If validation fails.

    """
    try:
        function_schemas: list[FunctionSchema] = [
            FunctionSchema.model_validate(function) for function in functions
        ]
    except ValidationError as e:
        raise ParsingException(str(e))

    return function_schemas


def get_boolean_automate() -> StringRouterState:
    """

    Get a state machine for boolean values.

    Returns:

        StringRouterState: The boolean state machine.

    """
    true_state = StaticStringState("true")
    false_state = StaticStringState("false")

    return StringRouterState(
        {true_state, false_state}, {true_state: [], false_state: []}
    )


def build_function_states(
    functions: list[dict[str, Any]]
) -> StringRouterState:
    """

    Build state machines for function definitions.

    Args:

        functions (list[dict[str, Any]]): List of function definitions.

    Returns:

        StringRouterState: The root state for function selection.

    """
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
            local_next_states.append(StaticStringState("}}"))
            routers.append(router)
            next_states[router] = local_next_states
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
            elif type.type == DataType.INTEGER:
                local_next_states.append(
                    IntegerState("}" if i == param_numbers - 1 else ",")
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


def build_prompts(
    prompts_path: str, functions: list[dict[str, Any]]
) -> list[Prompt]:
    """

    Build prompt objects from a JSON file.

    Args:

        prompts_path (str): Path to the prompts JSON file.

        functions (list[dict[str, Any]]): List of function definitions.

    Returns:

        list[Prompt]: List of prompt objects.

    Raises:

        ParsingException: If parsing fails.

    """
    prompts: list[Prompt] = []

    texts = load_json_file(prompts_path, "prompts")

    try:
        for text in texts:
            prompt = PromptSchema.model_validate(text)
            prompts.append(Prompt(prompt.prompt, functions))

        return prompts
    except ValidationError as e:
        raise ParsingException(f"Invalid prompt schema: {e}")


def build_state_machine(functions: list[dict[str, Any]]) -> StateMachine:
    """

    Build a state machine for function calling.

    Args:

        functions (list[dict[str, Any]]): List of function definitions.

    Returns:

        StateMachine: The state machine for constrained generation.

    """
    begin = StaticStringState("{\"name\":\"")
    router = build_function_states(functions)

    return StateMachine([begin, router])


def build_linked_state_machines(
    functions_definition_path: str,
    input_file_path: str,
) -> dict[Prompt, StateMachine]:
    """

    Build linked state machines for prompts and functions.

    Args:

        functions_definition_path (str): Path to functions definition JSON.

        input_file_path (str): Path to prompts JSON.

    Returns:

        dict[Prompt, StateMachine]: Mapping of prompts to state machines.

    """
    linked_state_machines: dict[Prompt, StateMachine] = {}

    funcs = load_json_file(functions_definition_path, "functions definition")

    prompts = build_prompts(input_file_path, funcs)
    for prompt in prompts:
        linked_state_machines[prompt] = build_state_machine(funcs)

    return linked_state_machines
