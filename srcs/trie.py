from llm_sdk import Small_LLM_Model
import json
from collections import deque
import functools


class TrieNode:
    def __init__(self) -> None:
        self.childrens: dict[str, TrieNode] = {}
        self.token: int | None = None


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()
        self.size = 0

    def insert(self, string: str, token_id: int) -> None:
        if not string:
            return
        cur = self.root

        for s in string:
            if s in cur.childrens:
                cur = cur.childrens[s]
            else:
                cur.childrens[s] = TrieNode()
                cur = cur.childrens[s]

        cur.token = token_id
        self.size += 1

    @functools.lru_cache()
    def search_prefixes(self, string: str) -> set[int]:
        cur = self.root

        result = []
        for s in string:
            if s in cur.childrens:
                cur = cur.childrens[s]
                if cur.token:
                    result.append(cur.token)
            else:
                break
        return set(result)

    @functools.lru_cache()
    def search_batch_prefixes(self, strings: tuple[str]) -> set[int]:
        result = []
        for string in strings:
            cur = self.root
            for s in string:
                if s in cur.childrens:
                    cur = cur.childrens[s]
                    if cur.token:
                        result.append(cur.token)
                else:
                    break
        return set(result)

    def constrained_search(self, constraint: str) -> set[int]:
        queue: deque[tuple[str, TrieNode]] = deque()

        for key, value in self.root.childrens.items():
            if key in constraint:
                queue.appendleft((key, value))

        result: list[int] = []

        while len(queue):
            next_key, node = queue.popleft()
            if node.token is not None:
                result.append(node.token)
            next = node
            for key, value in next.childrens.items():
                if key in constraint:
                    queue.appendleft((key, value))

        return set(result)

    def ended_search(
        self, constraint: str, escape: str, escape_char: str
    ) -> set[int]:
        queue: deque[tuple[str, TrieNode, bool]] = deque()
        result: list[int] = []

        for char, node in self.root.childrens.items():
            queue.append((char, node, False))

        while queue:
            char, node, is_escaped = queue.popleft()
            if char == escape_char and not is_escaped:
                if node.token is not None:
                    result.append(node.token)
                for next_char, next_node in node.childrens.items():
                    if next_char in escape:
                        queue.append((next_char, next_node, True))
            elif char == constraint and not is_escaped:
                if node.token is not None:
                    result.append(node.token)
            else:
                if node.token is not None:
                    result.append(node.token)
                for next_char, next_node in node.childrens.items():
                    queue.append((next_char, next_node, False))

        return set(result)


def get_token_trie(llm: Small_LLM_Model) -> Trie:
    trie = Trie()

    path = llm.get_path_to_tokenizer_file()
    with open(path, "r") as f:
        tokenizer_file = json.load(f)

    vocab: dict[str, int] = tokenizer_file["model"]["vocab"]
    for token in vocab.values():
        trie.insert(llm.decode(token), token)

    return trie
