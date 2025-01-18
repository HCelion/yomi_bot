from __future__ import annotations
import random
from abc import ABC
from typing import List, Union, Any

from collections.abc import MutableSequence


class Deck(MutableSequence, ABC):
    """
    Class for containing cards and card operations.
    A deck is a composition of between 0 to 52 cards.

    ...

    Attributes:
    ----------
    cards:
        A list of cards. May be empty.

    Methods
    -------
    __len__():
        Retrieve number of cards in deck.
    *item__():
        Various index based card operations.
    __add__():
        Combine two decks. 
    __repr__():
        Show cards in their current order.
    shuffle():
        Shuffle the cards in the deck.
    """
    def __init__(self, cards: List = []) -> None:
        self.cards = cards

    def __len__(self) -> int:
        return len(self.cards)

    def __add__(self, other: Deck) -> Deck:
        return Deck(cards=self.cards + other.cards)

    def __getitem__(self, index: Union[int, slice]) -> Any:
        return self.cards[index]

    def __setitem__(self, index: Union[int, slice], item: Any) -> None:
        self.cards[index] = item

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.cards[index]

    def __repr__(self) -> str:
        return self.cards.__repr__()

    def shuffle(self) -> None:
        random.shuffle(self.cards)
