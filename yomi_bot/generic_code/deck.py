import random
from abc import ABC

from collections.abc import MutableSequence


class Deck(MutableSequence, ABC):
    def __init__(self, cards=None):
        self.cards = cards

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

    def __add__(self, other):
        return Deck(cards=self.cards + other.cards)

    def __setitem__(self, index, item):
        self.cards[index] = item

    def __delitem__(self, index):
        del self.cards[index]

    def insert(self, index, item):
        self.cards[index] = item

    def __repr__(self):
        return self.cards.__repr__()

    def draw(self, n):
        if len(self.cards) >= n:
            removed_cards = [self.cards.pop() for _ in range(n)]
        else:
            removed_cards = [self.cards.pop() for _ in range(len(self.cards))]
        return Deck(removed_cards)

    def shuffle(self):
        random.shuffle(self.cards)

    def shuffled(self):
        self.shuffle()
        return self
