from yomi_bot.generic_code.player import Player
from yomi_bot.rps.rps_deck import RPSDeck
from yomi_bot.generic_code.deck import Deck
import random

class RPSPlayer(Player):
    
    deck_generator = RPSDeck
    
    def __init__(self, hand_size=3, strategy='best_play', side='left'):
        super().__init__(hand_size=hand_size, strategy=strategy, side=side)
        self.own_state['score'] = 0
        self.other_state['score'] = 0
        
    def choose_card(self):
        if self.strategy == 'random':
            return self.random_strategy()
        else:
            raise NotImplementedError('The player does not know the suggested strategy')
        
    def update_specific_state(self,state_update, state):
        state['score'] += state_update['score']
        for card in state_update['discards']:
            state['discard'][card] += 1
        
    def run_assigned_actions(self, action_updates):
        drawn_cards = self.deck.draw(action_updates['draw'])
        self.hand = self.hand + drawn_cards
        
