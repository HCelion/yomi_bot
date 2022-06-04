from yomi_bot.generic_code.deck import Deck
from itertools import product
from copy import deepcopy

rps_cards = [flavor+str(size) for flavor,size in product('RPS', [1,2,3])]

class RPSDeck(Deck):
    
    def __init__(self):
        self.cards = deepcopy(rps_cards)
        
