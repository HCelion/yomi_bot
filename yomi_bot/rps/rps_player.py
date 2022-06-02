from yomi_bot.generic_code.players import Player
from yomi_bot.rps.rps_deck import RPSDeck
import random

class RPSPlayer(Player):
    
    deck_generator = RPSDeck
    
    def __init__(self, hand_size=3, strategy='best_play'):
        super().__init__(hand_size=hand_size, strategy=strategy)
        self.own_state['score'] = 0
        self.other_state['score'] = 0
        
    def choose_card(self):
        if self.strategy == 'random':
            return self.random_strategy()
        else:
            raise NotImplementedError('The player does not know the suggested strategy')
        
        
    def update_state(self, update_dict):
        pass
        
