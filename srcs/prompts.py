from srcs.schemas import FunctionCallSchema


class Prompt:
    def __init__(self, prompt: str, funcs: str) -> None:
        schema = FunctionCallSchema.model_json_schema()
        self.initial_prompt = prompt
        self.final_prompt = (
            "<|im_start|>system\n"
            "You are a function calling AI model. You are provided with "
            "function signatures within <tools></tools> XML tags. You may "
            "call one or more functions to assist with the user query. "
            "Don't make assumptions about what values to plug into functions. "
            "Here are the available tools: "
            f"<tools> {funcs} </tools> "
            "Use the following pydantic model json schema for each tool "
            f"call you will make: {schema} "
            "For each function call return a json object with function name "
            "and parameters within <tool_call></tool_call> XML tags as "
            'follows: \n<tool_call>\n{"name": <function-name>, '
            '"parameters": <params-dict>}\n</tool_call><|im_end|>\n'
            "<|im_start|>user\n"
            f"{self.initial_prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<tool_call>\n"
        )

    def __str__(self) -> str:
        return self.final_prompt
