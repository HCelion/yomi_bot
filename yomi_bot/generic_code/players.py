from abc import ABC, abstractmethod
import random

class Player(ABC):
    
    def __init__(self, hand_size=3, strategy='best_play', side='left'):
        self.deck = self.deck_generator().shuffled()
        self.hand = self.deck.draw(hand_size)
        self.strategy = strategy
        self.side = side
        
        if side == 'left':
            self.other_side = 'right'
        else:
            self.other_side = 'left'
            
        self.own_state = {}
        self.other_state = {}
        self.own_state['discard'] = {card:0 for card in self.deck_generator()}
        self.other_state['discard'] = {card:0 for card in self.deck_generator()}
    
    @property
    @abstractmethod
    def deck_generator(self):
        pass

    @abstractmethod
    def choose_card(self):
        pass
    
    def update_state(self, update_dict):
        # Updates internal representations of self and other
        self.update_specific_state(update_dict[self.side]['state'], self.own_state)
        self.update_specific_state(update_dict[self.other_side]['state'], self.other_state)
    
        # Run assigned actions only for self
        self.run_assigned_actions(update_dict[self.side]['actions'])
    
    @abstractmethod
    def update_specific_state(self, state_update_dict, state):
        pass
        
    @abstractmethod
    def run_assigned_actions(self, action_updates):
        pass
        
    def random_strategy(self):
        if len(self.hand) > 0:
            random.shuffle(self.hand)
            return self.hand.pop()
        else:
            return None
