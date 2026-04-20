from srcs.machine import StateMachine
from srcs.states import StaticStringState, get_string_router_automate
from llm_sdk import Small_LLM_Model
from srcs.trie import get_token_trie
from srcs.utils import get_best_token
from srcs.parsing import build_function_states


class InferenceEngine:
    def __init__(self):
        self.llm = Small_LLM_Model()
        self.trie = get_token_trie(self.llm)
        begin = StaticStringState("{\"name\":\"")
        func_router = build_function_states(
            "data/input/functions_definition.json"
        )

        self.machine = StateMachine([begin, func_router])

        funcs = "fn_greet_user,fn_add_numbers,fn_reverse_string,fn_get_square_root,fn_send_money"
        messages = [
            {"role": "system", "content": "You are a ultra futuristic IT assistant which only responds a proper structured JSON"},
            {"role": "user", "content": f"Choisis une fonction pour dire bonjour a bcondemi into the following set: {funcs}"}
        ]

        prompt = self.llm._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        self.encoded_prompt = self.llm.encode(prompt)[0].tolist()

    def run(self) -> None:
        while not self.machine.is_done():
            logits = self.llm.get_logits_from_input_ids(self.encoded_prompt)
            authorized = self.machine.get_authorized_tokens(self.trie)
            best_token = get_best_token(authorized, logits)
            print(self.llm.decode(best_token), end="", flush=True)
            self.machine.consume(self.llm.decode(best_token))
            self.encoded_prompt.append(best_token)
