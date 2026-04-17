from collections import deque
from srcs.trie import Trie
from srcs.states import State, StateException


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

        try:
            self.states[0].consume(text)
        except StateException:
            raise StateMachineException

        if self.states[0].is_done():
            done_state = self.states.popleft()
            self.states.extendleft(reversed(done_state.get_next_states()))
            if not len(self.states):
                self.done = True

    def is_done(self) -> bool:
        return self.done

    def get_authorized_tokens(self, trie: Trie) -> set[int]:
        return self.states[0].get_valid_tokens(trie)
