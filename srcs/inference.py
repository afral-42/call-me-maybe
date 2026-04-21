import json

from srcs.machine import StateMachine
from srcs.states import StaticStringState
from llm_sdk import Small_LLM_Model
from srcs.trie import get_token_trie
from srcs.utils import get_best_token
from srcs.parsing import build_function_states


class InferenceEngine:  # Reste à implementer proprement le batching
    def __init__(self):
        self.llm = Small_LLM_Model()
        self.trie = get_token_trie(self.llm)
        self.result = []  # Nous utiliserons plus tard des numpy array
        begin = StaticStringState("{\"name\":\"")
        func_router = build_function_states(  # On prefera passer directement le fichier json chargé ce sera plus simple et on ouvrira ici avec une factory
            "data/input/functions_definition.json"
        )
        with open("data/input/functions_definition.json") as f:
            funcs = f.read()

        self.machine = StateMachine([begin, func_router]) # Il nous faut une factory des etats !
        # TODO factory du prompt
        prompt = f'''
<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> [{{"type": "function", "function": {{"name": "get_stock_fundamentals", "description": "Get fundamental data for a given stock symbol using yfinance API.", "parameters": {{"type": "object", "properties": {{"symbol": {{"type": "string"}}}}, "required": ["symbol"]}}}}}}] </tools> Use the following pydantic model json schema for each tool call you will make: {{"title": "FunctionCall", "type": "object", "properties": {{"name": {{"title": "Name", "type": "string"}}, "arguments": {{"title": "Arguments", "type": "object"}}}}, "required": ["name", "arguments"]}} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{funcs}
</tool_call><|im_end|>
<|im_start|>user
Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS
<|im_start|>assistant
<tool_call>
'''
        self.encoded_prompt = self.llm.encode(prompt)[0].tolist()

    def run(self) -> None:
        while not self.machine.is_done():
            logits = self.llm.get_logits_from_input_ids(self.encoded_prompt)
            authorized = self.machine.get_authorized_tokens(self.trie)
            best_token = get_best_token(authorized, logits)
            print(self.llm.decode(best_token), end="", flush=True)
            self.machine.consume(self.llm.decode(best_token))
            self.encoded_prompt.append(best_token)
            self.result.append(best_token)

    def save_answer(self) -> None:
        try:
            result_txt = self.llm.decode(self.result)
            print(result_txt)
            result = json.loads(result_txt)

            with open("result.json", "w") as f:
                json.dump(result, f)

        except json.JSONDecodeError:
            raise  # Il va falloir mettre des exceptions plus intelligentes
