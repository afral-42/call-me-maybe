from collections import deque
from src.trie import Trie
from src.states import State
from llm_sdk import Small_LLM_Model


class StateMachineException(Exception):
    """

    Exception raised for state machine errors.

    """
    pass


class StateMachine:
    """

    Finite state machine for constrained token generation.

    """

    def __init__(self, states: list[State]) -> None:
        """

        Initialize the state machine.

        Args:

            states (list[State]): List of states in the machine.

        """
        self.states = deque(states)
        self.done = False

    def consume(self, text: str) -> None:
        """

        Consume text and update the state machine.

        Args:

            text (str): The text to consume.

        """
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
        """

        Check if the state machine is done.

        Returns:

            bool: True if done, False otherwise.

        """
        return self.done

    def get_authorized_tokens(self, trie: Trie) -> set[int]:
        """

        Get the set of authorized token IDs from the current state.

        Args:

            trie (Trie): The token trie.

        Returns:

            set[int]: Set of authorized token IDs.

        """
        return self.states[0].get_valid_tokens(trie)

    def intercept_token(
        self, token_str: str, token_id: int, llm: Small_LLM_Model
    ) -> list[int]:
        """

        Intercept and potentially split a token.

        Args:

            token_str (str): The token string.

            token_id (int): The token ID.

            llm (Small_LLM_Model): The language model.

        Returns:

            list[int]: List of token IDs.

        """
        return self.states[0].intercept_token(token_str, token_id, llm)

    def get_static_string(self) -> str | None:
        """

        Get the static string from the current state if available.

        Returns:

            str | None: The static string or None.

        """
        return self.states[0].get_static_string()
