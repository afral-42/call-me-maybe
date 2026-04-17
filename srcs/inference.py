from srcs.machine import StateMachine
from srcs.states import StaticStringState, get_string_router_automate
from llm_sdk import Small_LLM_Model
from srcs.trie import get_token_trie
from srcs.utils import get_best_token


class InferenceEngine:
    def __init__(self):
        begin = StaticStringState("{\"name\":\"")
        string_router = get_string_router_automate([
            "fn_greet_user",
            "fn_add_numbers",
            "fn_reverse_string",
            "fn_get_square_root",
            "fn_send_money"
        ])

        self.machine = StateMachine([begin, string_router])
        self.llm = Small_LLM_Model()
        self.trie = get_token_trie(self.llm)

        funcs = "fn_greet_user,fn_add_numbers,fn_reverse_string,fn_get_square_root,fn_send_money"
        messages = [
            {"role": "system", "content": "You are a ultra futuristic IT assistant who can interact with the world"},
            {"role": "user", "content": f"Please, choose only one function to send money (1000$) the following set: {funcs}"}
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
