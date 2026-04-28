from typing import Any


class Prompt:
    """

    Represents a prompt for function calling with formatted functions.

    """

    def __init__(self, prompt: str, funcs: list[dict[str, Any]]) -> None:
        """

        Initialize the prompt.

        Args:

            prompt (str): The user prompt.

            funcs (list[dict[str, Any]]): List of function definitions.

        """
        self.initial_prompt = prompt
        self.final_prompt = (
            "<|im_start|>system\n"
            "You are a function calling AI model.\n"
            f"<tools> {self.format_functions_to_signatures(funcs)} </tools>\n"
            '<|im_end|>\n'
            "<|im_start|>user\n"
            f"{self.initial_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<tool_call>\n"
        )

    def format_functions_to_signatures(
        self, functions: list[dict[str, Any]]
    ) -> str:
        """

        Format function definitions into signatures.

        Args:

            functions (list[dict[str, Any]]): List of function definitions.

        Returns:

            str: Formatted signatures.

        """
        signatures = []
        for func in functions:
            name = func.get("name", "")
            desc = func.get("description", "")

            params = []
            for p_name, p_details in func.get("parameters", {}).items():
                p_type = p_details.get("type", "any")
                params.append(f"{p_name}: {p_type}")

            param_str = ", ".join(params)

            signatures.append(f"# {desc}\n{name}({param_str})")

        return "\n\n".join(signatures)

    def __str__(self) -> str:
        """

        Return the full prompt string.

        Returns:

            str: The formatted prompt.

        """
        return self.final_prompt
