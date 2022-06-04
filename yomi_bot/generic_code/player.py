from abc import ABC, abstractmethod
import random


class Player(ABC):

    # TODO Make player a runnable subprocess that listens in on the Arena for
    # instructions, so it can run in parallel to other players

    def __init__(self, hand_size=3, strategy="best_play", side="left"):
        self.deck = self.deck_generator().shuffled()
        self.hand = self.deck.draw(hand_size)
        self.strategy = strategy
        self.side = side

        if side == "left":
            self.other_side = "right"
        else:
            self.other_side = "left"

        self.own_state = self.generate_initial_state()
        self.other_state = self.generate_initial_state()

    @property
    @abstractmethod
    def deck_generator(self):
        pass

    @abstractmethod
    def generate_initial_state(self):
        ...

    def initialise_discard(self):
        return {card: 0 for card in self.deck_generator()}

    @abstractmethod
    def choose_card(self):
        pass

    def update_state(self, update_dict):
        # Updates internal representations of self and other
        self.update_specific_state(update_dict[self.side], self.own_state)
        self.update_specific_state(update_dict[self.other_side], self.other_state)

        # Run assigned actions only for self
        self.run_assigned_actions(update_dict[self.side])

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
