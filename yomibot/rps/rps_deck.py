from yomibot.generic_code.deck import Deck
from itertools import product
from copy import deepcopy
from typing import List

rps_cards: List[str] = [
    flavor + str(size) for flavor, size in product("RPS", [1, 2, 3])
]


class RPSDeck(Deck):
    def __init__(self):
        self.cards = deepcopy(rps_cards)
