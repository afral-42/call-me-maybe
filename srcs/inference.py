import json

from srcs.machine import StateMachine
from llm_sdk import Small_LLM_Model
from srcs.trie import Trie, get_token_trie
from srcs.prompts import Prompt
from typing import Self
from srcs.parsing import build_linked_state_machines
import numpy as np


class InferenceException(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Inference error: {detail}")


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

        self.built_prompts: dict[Prompt, list[int]] = {}
        self.results: dict[Prompt, str] = {
            prompt: "" for prompt in linked_state_machines
        }

        self._batch_encode()

    def _batch_encode(self) -> None:
        for prompt in self.machines.keys():
            self.built_prompts[prompt] = (
                (self.llm.encode(str(prompt)))[0].tolist()
            )

    def run(self) -> None:
        for prompt, ids in self.built_prompts.items():
            while not self.machines[prompt].is_done():
                authorized = self.machines[prompt].get_authorized_tokens(
                    self.trie
                )

                static_string = self.machines[prompt].get_static_string()
                if static_string is not None:
                    tokens: list[int] = (
                        self.llm.encode(static_string)[0].tolist()
                    )
                    for token in tokens:
                        self.built_prompts[prompt].append(token)
                    self.results[prompt] = (
                        f"{self.results[prompt]}{static_string}"
                    )
                    self.machines[prompt].consume(static_string)
                    continue

                elif len(authorized) == 1:
                    best_token = list(authorized)[0]

                else:
                    logits = self.llm.get_logits_from_input_ids(ids)
                    best_token = self._get_best_token(authorized, logits)

                best_token_str = self.llm.decode([best_token])

                cleaned_tokens = self.machines[prompt].intercept_token(
                    best_token_str, best_token, self.llm
                )
                for token in cleaned_tokens:
                    token_str = self.llm.decode([token])
                    self.results[prompt] = (
                        f"{self.results[prompt]}{token_str}"
                    )
                    self.built_prompts[prompt].append(token)
                    self.machines[prompt].consume(token_str)

    def save_results(self) -> None:
        final_output = []

        for prompt, result_str in self.results.items():
            try:
                parsed_json = json.loads(result_str)
                parsed_json["prompt"] = prompt.initial_prompt
                final_output.append(parsed_json)
            except json.JSONDecodeError:
                final_output.append(
                    {"error": "Invalid JSON", "raw": result_str}
                )

        try:
            with open(self.output_file_path, "w") as f:
                json.dump(final_output, f)
        except IOError as e:
            pass  # Je sais pas quoi faire encore

    def _get_best_token(self, authorized: set[int], logits: list[float]):
        if len(authorized) > 1000:
            return int(np.argmax(logits))

        return max(authorized, key=lambda i: logits[i])


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
