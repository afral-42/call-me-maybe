import json

from srcs.machine import StateMachine
from srcs.states import StaticStringState
from llm_sdk import Small_LLM_Model
from srcs.trie import Trie, get_token_trie
from srcs.parsing import build_function_states
from srcs.prompts import Prompt
from typing import Self
from parsing import build_linked_state_machines



def get_best_token(authorized: set[int], logits: list[float]):
    return max(authorized, key=lambda i: logits[i])


class InferenceEngine:
    def __init__(self):
        self.llm = Small_LLM_Model()
        self.trie = get_token_trie(self.llm)
        self.result = []
        begin = StaticStringState("{\"name\":\"")
        func_router = build_function_states(  # On prefera passer directement le fichier json chargé ce sera plus simple et on ouvrira ici avec une factory
            "data/input/functions_definition.json"
        )
        with open("data/input/functions_definition.json") as f:
            funcs = f.read()

        self.machine = StateMachine([begin, func_router]) # Il nous faut une factory des etats !
        prompt = Prompt("Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'", funcs)
        #print(str(prompt))
        self.encoded_prompt = self.llm.encode(str(prompt))[0].tolist()

    def run(self) -> None:
        while not self.machine.is_done():
            authorized = self.machine.get_authorized_tokens(self.trie)

            if len(authorized) == 1:
                best_token = list(authorized)[0]
                best_token_str = self.llm.decode(best_token)

                self.result.append(best_token)
                self.machine.consume(best_token_str)
                self.encoded_prompt.append(best_token)

            else:
                logits = self.llm.get_logits_from_input_ids(
                    self.encoded_prompt
                )
                selected_token = get_best_token(authorized, logits)

                selected_token_str = self.llm.decode(selected_token)
                best_tokens = self.machine.intercept_token(
                    selected_token_str, selected_token, self.llm
                )

                for best_token in best_tokens:
                    best_token_str = self.llm.decode(best_token)
                    #print(self.llm.decode(best_token), end="", flush=True)
                    self.machine.consume(best_token_str)
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
            raise


class BatchInferenceEngine:
    def __init__(
        self,
        llm: Small_LLM_Model,
        trie: Trie,
        output_file_path: str,
        linked_state_machines: dict[Prompt, StateMachine]
    ) -> None:
        self.machines = linked_state_machines
        self.trie = trie
        self.llm = llm
        self.output_file_path = output_file_path

        self.built_prompts: dict[Prompt, list[int]]
        self.results = dict[Prompt, str]

    def batch_encode(self) -> None:
        for prompt in self.machines.keys():
            self.built_prompts[prompt] = (
                (self.llm.encode(str(prompt)))[0].tolist()
            )

    def run(self) -> None:
        pass

    def save_results(self) -> None:
        pass

    @classmethod
    def build_engine(
        cls,
        input_path: str,
        functions_definition_path: str,
        output_path: str
    ) -> Self:
        llm = Small_LLM_Model()
        trie = get_token_trie(llm)
        machines = build_linked_state_machines(
            functions_definition_path, input_path
        )

        return cls(
            llm=llm,
            trie=trie,
            output_file_path=output_path,
            linked_state_machines=machines
        )
