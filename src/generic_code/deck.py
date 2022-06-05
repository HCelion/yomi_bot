from __future__ import annotations
import random
from abc import ABC
from typing import List, Union, Any

from collections.abc import MutableSequence


class Deck(MutableSequence, ABC):
    def __init__(self, cards: List) -> None:
        self.cards = cards

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, index: Union[int, slice]) -> Any:
        return self.cards[index]

    def __add__(self, other: Deck) -> Deck:
        return Deck(cards=self.cards + other.cards)

    def __setitem__(self, index: Union[int, slice], item: Any) -> None:
        self.cards[index] = item

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.cards[index]

    def insert(self, index: int, item: Any) -> None:
        self.cards[index] = item

    def __repr__(self) -> str:
        return self.cards.__repr__()

    def draw(self, n: int) -> Deck:
        if len(self.cards) >= n:
            removed_cards = [self.cards.pop() for _ in range(n)]
        else:
            removed_cards = [self.cards.pop() for _ in range(len(self.cards))]
        return Deck(removed_cards)

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def shuffled(self) -> Deck:
        self.shuffle()
        return self
