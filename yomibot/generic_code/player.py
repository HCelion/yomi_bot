from abc import ABC, abstractmethod
import random


class Player(composition, ABC):
    """
    Class for containing decks and deck operations.
    A player is a composition of 3 decks with some strategy on top.

    ...

    Attributes:
    ----------
    deck_composition:
        The character deck, minus 1 throw, 2 blocks and 1 burst.
    hand_size:
        The amount of cards that the character draws on initialisation.
    strategy:
        How the character will be played.

    Methods
    -------
    draw():
        If the amount of cards in the Deck:
            - Greater than 0: pop a card into the character hand.
            - Exactly 0:
                1. Set the thrash as the deck
                2. Shuffle the deck and empty the trash.
            - Less than 0: throw an error.
    random_strategy():
        Shuffle the hand, play a popped card.
    """

    def __init__(
        self,
        deck_composition,
        hand_size: int = 3,
        strategy: str = "best_play",
    ):
        self.hand = Deck({"throw": 1, "blocks": 2, "burst": 1})
        self.deck = Deck([Card(card) for card in cards])
        self.thrash = Deck()
        self.deck.shuffle()  # Could have made part of deck init but not worth it

        self.strategy = strategy
        self.side = side

    def draw(self):
        deck_len = len(self.deck)
        if deck_len >= 0:
            self.hand.cards.append(self.deck.cards.pop())
        elif deck_len == 0:
            self.deck = self.thrash.shuffle()
            self.trash = Deck()
        elif deck_len < 0:
            raise ("Deck len should never be less than zero!")

    def random_strategy(self):
        if len(self.hand) > 0:
            random.shuffle(self.hand)
            return self.hand.pop()
        else:
            return None
