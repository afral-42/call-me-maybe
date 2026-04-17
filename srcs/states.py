from abc import ABC, abstractmethod
from srcs.trie import Trie
from enum import Enum, auto


class StateException(Exception):
    pass


class State(ABC):
    @abstractmethod
    def consume(self, text: str) -> str:
        """Méthode pour passer à l'état suivant en fonction de la donnée"""
        pass

    @abstractmethod
    def get_valid_tokens(self, trie: Trie) -> set[int]:
        """Méthode pour obtenir la liste des tokens valides en fonction de l'état"""
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Méthode pour savoir si un état est terminé ou non"""

    def get_next_states(self) -> list["State"]:
        return []


class StaticStringState(State):
    def __init__(self, string: str) -> None:
        self.string = string

    def consume(self, text: str) -> str:
        # Il faudra bien sur checker que c'est juste et si non raise 
        # Meme si théoriquement les logits ont ete bloqués
        if self.string.startswith(text):
            self.string = self.string[len(text):]
        else:
            raise StateException
        return ""  # On verra après ;)

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        return trie.search_prefixes(self.string)

    def is_done(self) -> bool:
        if self.string == "":
            return True
        return False


class NumberMachineState(Enum):
    SIGN = auto()
    START = auto()
    INTEGRAL = auto()
    POINT = auto()
    FRACTIONAL = auto()


class NumberCharType(Enum):
    DIGIT_WITH_ZERO = "0123456789"
    DIGIT = "123456789"
    POINT = "."
    END = ","
    MINUS = "-"
    ZERO = "0"
    INVALID = "!"


class NumberState(State):
    def __init__(self, trie: Trie) -> None:
        self.done = False
        self.state = NumberMachineState.SIGN

        self.authorized = {
            NumberMachineState.SIGN: trie.constrained_search("-0123456789"),
            NumberMachineState.START: trie.constrained_search("0123456789"),
            NumberMachineState.INTEGRAL: trie.constrained_search("0123456789."),  # Attention car l'arbre peut tres bien renvoyer .. si prefix! (ajouter un argument "unique")
            NumberMachineState.POINT: trie.constrained_search("0123456789"),
            NumberMachineState.FRACTIONAL: trie.constrained_search("0123456789")
        }

        self.transitions = {
            (NumberMachineState.SIGN, NumberCharType.MINUS): (
                NumberMachineState.START
            ),
            (NumberMachineState.SIGN, NumberCharType.DIGIT): (
                NumberMachineState.INTEGRAL
            ),
            (NumberMachineState.SIGN, NumberCharType.ZERO): (
                NumberMachineState.POINT
            ),
            (NumberMachineState.INTEGRAL, NumberCharType.DIGIT_WITH_ZERO): (
                NumberMachineState.INTEGRAL
            ),
            (NumberMachineState.INTEGRAL, NumberCharType.POINT): (
                NumberMachineState.POINT
            ),
            (NumberMachineState.POINT, NumberCharType.DIGIT_WITH_ZERO): (
                NumberMachineState.FRACTIONAL
            ),
        }

    def _get_char_type(self, char: str) -> NumberCharType:
        for key in NumberCharType:
            if char in key.value:
                return key

        return NumberCharType.INVALID

    def consume(self, text: str) -> str:
        for i, char in enumerate(text):
            char_type = self._get_char_type(char)

            match char_type:
                case NumberCharType.INVALID:
                    raise StateException
                case NumberCharType.END:
                    self.done = True
                    return text[i:]
                case _:
                    self.state = self.transitions[(self.state, char_type)]

        return ""

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        return self.authorized[self.state]

    def is_done(self) -> bool:
        return self.done


class DynamicStringState(State):
    pass


class StringRouterState(State):
    def __init__(
        self,
        options: set[StaticStringState],
        next_states: dict[StaticStringState, list[State]]
    ) -> None:
        self.options = options
        self.done = False
        self.next_states = next_states

    def consume(self, text: str) -> str:
        if not len(self.options):
            raise StateException
        to_remove = []

        for state in self.options:
            try:
                state.consume(text)
            except StateException:
                to_remove.append(state)
            if state.is_done():
                self.done = True
                self.winner = state

        for state in to_remove:
            self.options.remove(state)

        return ""  # On verra après ;)

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        return trie.search_batch_prefixes(
            [state.string for state in self.options]
        )

    def is_done(self) -> bool:
        return self.done

    def get_next_states(self) -> list[State]:
        return self.next_states[self.winner]


def get_string_router_automate(options: list[str]) -> StringRouterState:
    functions = [StaticStringState(option) for option in options]
    next_states = {}

    for function in functions:
        next_states[function] = [StaticStringState("\",\"parameters\":{"), NumberState()]

    return StringRouterState(set(functions), next_states)
