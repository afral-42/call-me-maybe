from abc import ABC, abstractmethod
from src.trie import Trie
from enum import Enum, auto
from llm_sdk import Small_LLM_Model


class StateException(Exception):
    def __init__(self, detail: str) -> None:
        """

        Initialize the state exception.

        Args:

            detail (str): The detail message for the error.

        """
        super().__init__(f"Generation error: {detail}")


class State(ABC):
    """

    Abstract base class for states in the state machine.

    """

    @abstractmethod
    def consume(self, text: str) -> None:
        """

        Consume text and update the state.

        Args:

            text (str): The text to consume.

        """
        pass

    @abstractmethod
    def get_valid_tokens(self, trie: Trie) -> set[int]:
        """

        Get valid token IDs for the current state.

        Args:

            trie (Trie): The token trie.

        Returns:

            set[int]: Set of valid token IDs.

        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """

        Check if the state is done.

        Returns:

            bool: True if done.

        """
        pass

    def get_next_states(self) -> list["State"]:
        """

        Get the next states after this state is done.

        Returns:

            list[State]: List of next states.

        """
        return []

    def intercept_token(
        self, token_str: str, token_id: int, llm: Small_LLM_Model
    ) -> list[int]:
        """

        Intercept a token, potentially splitting it.

        Args:

            token_str (str): The token string.

            token_id (int): The token ID.

            llm (Small_LLM_Model): The language model.

        Returns:

            list[int]: List of token IDs.

        """
        return [token_id]

    def get_static_string(self) -> str | None:
        """

        Get a static string if available.

        Returns:

            str | None: The static string or None.

        """
        return None


class StaticStringState(State):
    """

    State that matches a static string.

    """

    def __init__(self, string: str) -> None:
        """

        Initialize the static string state.

        Args:

            string (str): The string to match.

        """
        self.string = string

    def consume(self, text: str) -> None:
        """

        Consume text by removing prefix.

        Args:

            text (str): The text to consume.

        Raises:

            StateException: If text doesn't match.

        """
        if self.string.startswith(text):
            self.string = self.string[len(text):]
        else:
            raise StateException(
                f"Expected to consume '{text}' in static string, "
                f"but remaining is '{self.string}'"
            )

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        """

        Get valid tokens for the remaining string.

        Args:

            trie (Trie): The token trie.

        Returns:

            set[int]: Set of valid token IDs.

        """
        return trie.search_prefixes(self.string)

    def is_done(self) -> bool:
        """

        Check if the string is fully consumed.

        Returns:

            bool: True if done.

        """
        if self.string == "":
            return True
        return False

    def get_static_string(self) -> str | None:
        """

        Get the remaining static string.

        Returns:

            str | None: The remaining string.

        """
        return self.string


class NumberMachineState(Enum):
    ZERO = auto()
    SIGN = auto()
    START_MINUS = auto()
    INTEGRAL = auto()
    POINT = auto()
    FRACTIONAL = auto()
    EXPONENT = auto()
    EXPONENT_SIGN = auto()
    EXPONENT_INTEGRAL = auto()


class NumberCharType(Enum):
    DIGITS = "123456789"
    POINT = "."
    END = ","
    MINUS = "-"
    ZERO = "0"
    EXPONENT = "eE"
    PLUS = "+"


class NumberState(State):
    """

    State for parsing JSON number values.

    """

    def __init__(self, end_char: str, max_length: int = 50) -> None:
        """

        Initialize the number state.

        Args:

            end_char (str): The character that ends the number.

            max_length (int): Maximum length allowed.

        """
        self.done = False
        self.state = NumberMachineState.SIGN
        self.len_count = 0
        self.max_length = max_length

        self.end = end_char

        self.search_map = {
            NumberMachineState.SIGN: ("-0123456789", "-"),
            NumberMachineState.ZERO: (f".{self.end}", ","),
            NumberMachineState.START_MINUS: ("0123456789", ""),
            NumberMachineState.INTEGRAL: (f"0123456789.{self.end}eE", "."),
            NumberMachineState.POINT: ("0123456789", ""),
            NumberMachineState.FRACTIONAL: (f"0123456789{self.end}eE", ","),
            NumberMachineState.EXPONENT: ("-+0123456789", "-+"),
            NumberMachineState.EXPONENT_SIGN: ("0123456789", ""),
            NumberMachineState.EXPONENT_INTEGRAL: (
                f"0123456789{self.end}", ","
            ),
        }

        self.transitions = {
            (NumberMachineState.SIGN, NumberCharType.MINUS): (
                NumberMachineState.START_MINUS
            ),
            (NumberMachineState.SIGN, NumberCharType.DIGITS): (
                NumberMachineState.INTEGRAL
            ),
            (NumberMachineState.SIGN, NumberCharType.ZERO): (
                NumberMachineState.ZERO
            ),
            (NumberMachineState.START_MINUS, NumberCharType.ZERO): (
                NumberMachineState.ZERO
            ),
            (NumberMachineState.START_MINUS, NumberCharType.DIGITS): (
                NumberMachineState.INTEGRAL
            ),
            (NumberMachineState.ZERO, NumberCharType.POINT): (
                NumberMachineState.POINT
            ),
            (NumberMachineState.INTEGRAL, NumberCharType.DIGITS): (
                NumberMachineState.INTEGRAL
            ),
            (NumberMachineState.INTEGRAL, NumberCharType.ZERO): (
                NumberMachineState.INTEGRAL
            ),
            (NumberMachineState.INTEGRAL, NumberCharType.POINT): (
                NumberMachineState.POINT
            ),
            (NumberMachineState.POINT, NumberCharType.DIGITS): (
                NumberMachineState.FRACTIONAL
            ),
            (NumberMachineState.POINT, NumberCharType.ZERO): (
                NumberMachineState.FRACTIONAL
            ),
            (NumberMachineState.FRACTIONAL, NumberCharType.ZERO): (
                NumberMachineState.FRACTIONAL
            ),
            (NumberMachineState.FRACTIONAL, NumberCharType.DIGITS): (
                NumberMachineState.FRACTIONAL
            ),
            (NumberMachineState.INTEGRAL, NumberCharType.EXPONENT): (
                NumberMachineState.EXPONENT
            ),
            (NumberMachineState.FRACTIONAL, NumberCharType.EXPONENT): (
                NumberMachineState.EXPONENT
            ),
            (NumberMachineState.EXPONENT, NumberCharType.MINUS): (
                NumberMachineState.EXPONENT_SIGN
            ),
            (NumberMachineState.EXPONENT, NumberCharType.PLUS): (
                NumberMachineState.EXPONENT_SIGN
            ),
            (NumberMachineState.EXPONENT, NumberCharType.DIGITS): (
                NumberMachineState.EXPONENT_INTEGRAL
            ),
            (NumberMachineState.EXPONENT, NumberCharType.ZERO): (
                NumberMachineState.EXPONENT_INTEGRAL
            ),
            (NumberMachineState.EXPONENT_SIGN, NumberCharType.DIGITS): (
                NumberMachineState.EXPONENT_INTEGRAL
            ),
            (NumberMachineState.EXPONENT_SIGN, NumberCharType.ZERO): (
                NumberMachineState.EXPONENT_INTEGRAL
            ),
            (NumberMachineState.EXPONENT_INTEGRAL, NumberCharType.DIGITS): (
                NumberMachineState.EXPONENT_INTEGRAL
            ),
            (NumberMachineState.EXPONENT_INTEGRAL, NumberCharType.ZERO): (
                NumberMachineState.EXPONENT_INTEGRAL
            ),
        }

    def _get_char_type(self, char: str) -> NumberCharType | str:
        """

        Get the type of a character for number parsing.

        Args:

            char (str): The character.

        Returns:

            NumberCharType | str: The type or the end char.

        Raises:

            StateException: If invalid character.

        """
        if char == self.end:
            return self.end

        for key in NumberCharType:
            if char in key.value:
                return key

        raise StateException(
            f"Invalid character '{char}' encountered during number parsing."
        )

    def consume(self, text: str) -> None:
        self.len_count += len(text)

        for i, char in enumerate(text):
            char_type = self._get_char_type(char)

            match char_type:
                case self.end:
                    self.done = True
                    return
                case _:
                    try:
                        if isinstance(char_type, NumberCharType):
                            self.state = self.transitions[
                                (self.state, char_type)
                            ]
                    except KeyError:
                        raise StateException(
                            f"Invalid transition from state {self.state.name} "
                            f"with character '{char}'."
                        )

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        if self.len_count >= self.max_length:
            return trie.search_prefixes(self.end)

        search, unique = self.search_map[self.state]

        return trie.constrained_search_with_uniques(
            search,
            unique,
        )

    def is_done(self) -> bool:
        return self.done

    def intercept_token(
        self,
        token_str: str,
        token_id: int,
        llm: Small_LLM_Model
    ) -> list[int]:
        for i, c in enumerate(token_str):
            if (c == self.end):
                if i == len(token_str) - 1:
                    return [token_id]
                return (
                    llm.encode(token_str[:i])[0].tolist() +
                    llm.encode(self.end)[0].tolist()
                )
        return [token_id]


class IntegerMachineState(Enum):
    SIGN = auto()
    START_MINUS = auto()
    ZERO = auto()
    INTEGRAL = auto()


class IntegerCharType(Enum):
    DIGITS = "123456789"
    MINUS = "-"
    ZERO = "0"


class IntegerState(State):
    """

    State for parsing JSON integer values.

    """

    def __init__(self, end_char: str, max_length: int = 50) -> None:
        """

        Initialize the integer state.

        Args:

            end_char (str): The character that ends the integer.

            max_length (int): Maximum length allowed.

        """
        self.done = False
        self.state = IntegerMachineState.SIGN
        self.end = end_char
        self.max_length = max_length
        self.len_count = 0

        self.search_map = {
            IntegerMachineState.SIGN: ("-0123456789", "-"),
            IntegerMachineState.START_MINUS: ("0123456789", ""),
            IntegerMachineState.ZERO: (f"{self.end}", ""),
            IntegerMachineState.INTEGRAL: (f"0123456789{self.end}", ""),
        }

        self.transitions = {
            (IntegerMachineState.SIGN, IntegerCharType.MINUS): (
                IntegerMachineState.START_MINUS
            ),
            (IntegerMachineState.SIGN, IntegerCharType.DIGITS): (
                IntegerMachineState.INTEGRAL
            ),
            (IntegerMachineState.SIGN, IntegerCharType.ZERO): (
                IntegerMachineState.ZERO
            ),
            (IntegerMachineState.START_MINUS, IntegerCharType.ZERO): (
                IntegerMachineState.ZERO
            ),
            (IntegerMachineState.START_MINUS, IntegerCharType.DIGITS): (
                IntegerMachineState.INTEGRAL
            ),
            (IntegerMachineState.INTEGRAL, IntegerCharType.DIGITS): (
                IntegerMachineState.INTEGRAL
            ),
            (IntegerMachineState.INTEGRAL, IntegerCharType.ZERO): (
                IntegerMachineState.INTEGRAL
            ),
        }

    def _get_char_type(self, char: str) -> IntegerCharType | str:
        """

        Get the type of a character for integer parsing.

        Args:

            char (str): The character.

        Returns:

            IntegerCharType | str: The type or the end char.

        Raises:

            StateException: If invalid character.

        """
        if char == self.end:
            return self.end

        for key in IntegerCharType:
            if char in key.value:
                return key

        raise StateException(
            f"Invalid character '{char}' encountered during integer parsing."
        )

    def consume(self, text: str) -> None:
        self.len_count += len(text)

        for i, char in enumerate(text):
            char_type = self._get_char_type(char)

            match char_type:
                case self.end:
                    self.done = True
                    return
                case _:
                    try:
                        if isinstance(char_type, IntegerCharType):
                            self.state = self.transitions[
                                (self.state, char_type)
                            ]
                    except KeyError:
                        raise StateException(
                            f"Invalid transition from state {self.state.name} "
                            f"with character '{char}'."
                        )

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        if self.len_count >= self.max_length:
            return trie.search_prefixes(self.end)

        search, unique = self.search_map[self.state]

        return trie.constrained_search_with_uniques(
            search,
            unique,
        )

    def is_done(self) -> bool:
        return self.done

    def intercept_token(
        self,
        token_str: str,
        token_id: int,
        llm: Small_LLM_Model
    ) -> list[int]:
        for i, c in enumerate(token_str):
            if c == self.end:
                if i == len(token_str) - 1:
                    return [token_id]
                return (
                    llm.encode(token_str[:i])[0].tolist() +
                    llm.encode(self.end)[0].tolist()
                )
        return [token_id]


class StringMachineState(Enum):
    ESCAPE = auto()
    NORMAL = auto()


class StringCharType(Enum):
    ESCAPE_FLAG = "\\"
    OTHER = ""
    END = "\""


class DynamicStringState(State):
    """

    State for parsing JSON string values with escaping.

    """

    def __init__(self, max_length: int = 250) -> None:
        """

        Initialize the dynamic string state.

        Args:

            max_length (int): Maximum length allowed.

        """
        self.state = StringMachineState.NORMAL
        self.done = False
        self.all_tokens: set[int] | None = None
        self.max_length = max_length
        self.len_count = 0

        self.transitions = {
            (StringMachineState.ESCAPE, StringCharType.OTHER): (
                StringMachineState.NORMAL
            ),
            (StringMachineState.NORMAL, StringCharType.OTHER): (
                StringMachineState.NORMAL
            ),
            (StringMachineState.NORMAL, StringCharType.ESCAPE_FLAG): (
                StringMachineState.ESCAPE
            ),
            (StringMachineState.ESCAPE, StringCharType.ESCAPE_FLAG): (
                StringMachineState.NORMAL
            ),
        }

    def _get_char_type(self, char: str) -> StringCharType:
        """

        Get the type of a character for string parsing.

        Args:

            char (str): The character.

        Returns:

            StringCharType: The type.

        """
        for key in StringCharType:
            if char in key.value:
                return key

        return StringCharType.OTHER

    def consume(self, text: str) -> None:
        self.len_count += len(text)

        for i, char in enumerate(text):
            char_type = self._get_char_type(char)

            match char_type:
                case StringCharType.END:
                    if self.state == StringMachineState.ESCAPE:
                        self.state = StringMachineState.NORMAL
                    else:
                        self.done = True
                        return
                case _:
                    try:
                        self.state = self.transitions[(self.state, char_type)]
                    except KeyError:
                        raise StateException(
                            f"Invalid transition from state {self.state.name} "
                            f"with character '{char}' in dynamic string."
                        )

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        """

        Get valid tokens for string parsing.

        Args:

            trie (Trie): The token trie.

        Returns:

            set[int]: Set of valid token IDs.

        """
        if self.state == StringMachineState.ESCAPE:
            return trie.constrained_search("\"\\/bfnrtu")
        if self.len_count >= self.max_length:
            return trie.search_prefixes("\"")
        if not self.all_tokens:
            self.all_tokens = set(range(trie.size))
        return self.all_tokens

    def is_done(self) -> bool:
        """

        Check if string parsing is done.

        Returns:

            bool: True if done.

        """
        return self.done

    def intercept_token(
        self,
        token_str: str,
        token_id: int,
        llm: Small_LLM_Model
    ) -> list[int]:
        """

        Intercept token for string parsing.

        Args:

            token_str (str): The token string.

            token_id (int): The token ID.

            llm (Small_LLM_Model): The language model.

        Returns:

            list[int]: List of token IDs.

        """
        escaped = (self.state == StringMachineState.ESCAPE)

        for i, c in enumerate(token_str):
            if c == "\\":
                escaped = True
            elif (c == "\"" and not escaped):
                if i == len(token_str) - 1:
                    return [token_id]
                return (
                    llm.encode(token_str[:i])[0].tolist() +
                    llm.encode("\"")[0].tolist()
                )
            elif escaped:
                escaped = False

        return [token_id]


class StringRouterState(State):
    """

    State that routes to different options based on string matching.

    """

    def __init__(
        self,
        options: set[StaticStringState],
        next_states: dict[StaticStringState, list[State]]
    ) -> None:
        """

        Initialize the string router state.

        Args:

            options (set[StaticStringState]): Set of option states.

            next_states (dict[StaticStringState, list[State]]):
                Next states for each option.

        """
        self.options = options
        self.done = False
        self.next_states = next_states

    def consume(self, text: str) -> None:
        """

        Consume text and narrow down options.

        Args:

            text (str): The text to consume.

        Raises:

            StateException: If no options remain.

        """
        if not len(self.options):
            raise StateException(
                f"Router lost: No valid function options "
                f"remaining for input '{text}'."
            )
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

    def get_valid_tokens(self, trie: Trie) -> set[int]:
        """

        Get valid tokens for the remaining options.

        Args:

            trie (Trie): The token trie.

        Returns:

            set[int]: Set of valid token IDs.

        """
        return trie.search_batch_prefixes(
            tuple([state.string for state in self.options])
        )

    def is_done(self) -> bool:
        """

        Check if routing is done.

        Returns:

            bool: True if done.

        """
        return self.done

    def get_next_states(self) -> list[State]:
        """

        Get the next states after routing.

        Returns:

            list[State]: List of next states.

        """
        return self.next_states[self.winner]

    def get_static_string(self) -> str | None:
        """

        Get static string if only one option.

        Returns:

            str | None: The static string or None.

        """
        if len(self.options) == 1:
            return list(self.options)[0].get_static_string()
        return None
