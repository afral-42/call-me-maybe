class Prompt:
    def __init__(self, prompt: str, funcs: list[dict]) -> None:
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
        self, functions: list[dict]
    ) -> str:
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
        return self.final_prompt
