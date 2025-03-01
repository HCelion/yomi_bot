from yomibot.generic_code.deck import Deck
from itertools import product
from copy import deepcopy
from typing import List

from yomibot.generic_code.cards import Card

rps_cards: List[str] = [
    Card(str(speed), str(damage), flavor)
    for flavor, speed, damage in product("ADT", [1, 2, 3], [1, 2, 3])
]


class RPSDeck(Deck):
    def __init__(self):
        self.cards = deepcopy(rps_cards)
