from abc import ABC, abstractmethod
import random

class Player(ABC):
    
    def __init__(self, hand_size=3, strategy='best_play'):
        self.deck = self.deck_generator().shuffled()
        self.hand = self.deck.draw(hand_size)
        self.strategy = strategy
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
    
    @abstractmethod
    def update_state(self, update_dict):
        pass
        
    def random_strategy(self):
        if len(self.hand) > 0:
            random.shuffle(self.hand)
            return self.hand.pop()
        else:
            return None
