from yomi_bot.generic_code.player import Player
from yomi_bot.rps.rps_deck import RPSDeck, rps_cards
import numpy as np


class RPSPlayer(Player):

    deck_generator = RPSDeck

    def __init__(self, hand_size=3, strategy="best_play", side="left"):
        super().__init__(hand_size=hand_size, strategy=strategy, side=side)

    def generate_initial_state(self):
        state = {}
        state["score"] = 0
        state["hand_size"] = 3
        state["deck_size"] = 6
        state["discard"] = self.initialise_discard()
        return state

    def choose_card(self):
        if self.strategy == "random":
            return self.random_strategy()
        else:
            raise NotImplementedError("The player does not know the suggested strategy")

    def update_specific_state(self, update_dict, state):
        state_update = update_dict["state"]
        action_update = update_dict["actions"]
        state["score"] += state_update["score"]
        for card in state_update["discards"]:
            state["discard"][card] += 1

        num_draws = action_update["draw"]
        num_discards = len(state_update["discards"])
        actual_draws = min(num_draws, state["deck_size"])

        # Alternatively we could have them as @properties for direct lookups
        # However, that would mean we have to have access to the other players hand size, which could be problematic once the players are in separate processes
        # The explicit counting, while cumbersome, might be preferred here
        state["deck_size"] = state["deck_size"] - actual_draws
        state["hand_size"] = state["hand_size"] + actual_draws - num_discards

    def run_assigned_actions(self, update_dict):
        drawn_cards = self.deck.draw(update_dict["actions"]["draw"])
        self.hand = self.hand + drawn_cards

    @staticmethod
    def sample_hand(hand_size, discard, num_samples):
        """Does random sampling sampling with equal probabilities"""
        # Only sample from cards that are not already in the discard

        cards_to_sample = [card for card in rps_cards if discard[card] == 0]
        sample_array = np.full((num_samples, hand_size), "aa")

        for i in range(num_samples):
            sample_array[i, :] = np.random.choice(cards_to_sample, size=hand_size)

        return sample_array
