from collections import deque
from src.trie import Trie
from src.states import State
from llm_sdk import Small_LLM_Model


class StateMachineException(Exception):
    pass


class StateMachine:
    def __init__(self, states: list[State]) -> None:
        self.states = deque(states)
        self.done = False

    def consume(self, text: str) -> None:
        if not len(self.states):
            self.done = True
            return

        self.states[0].consume(text)

        if self.states[0].is_done():
            done_state = self.states.popleft()
            self.states.extendleft(reversed(done_state.get_next_states()))
            if not len(self.states):
                self.done = True

    def is_done(self) -> bool:
        return self.done

    def get_authorized_tokens(self, trie: Trie) -> set[int]:
        return self.states[0].get_valid_tokens(trie)

    def intercept_token(
        self, token_str: str, token_id: int, llm: Small_LLM_Model
    ) -> list[int]:
        return self.states[0].intercept_token(token_str, token_id, llm)

    def get_static_string(self) -> str | None:
        return self.states[0].get_static_string()
