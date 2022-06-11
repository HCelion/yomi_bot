from src.generic_code.player import Player
from src.rps.rps_deck import RPSDeck, rps_cards
import src.rps.rps_rules as rps
import numpy as np
from typing import Iterable
import pygambit


class RPSPlayer(Player):

    deck_generator = RPSDeck

    def __init__(
        self, hand_size: int = 3, strategy: str = "best_play", side: str = "left"
    ):
        super().__init__(hand_size=hand_size, strategy=strategy, side=side)
        self.payoff_lookup = rps.generate_rps_payoff_lookup()

    def generate_initial_state(self) -> dict:
        state = {}
        state["score"] = 0
        state["hand_size"] = 3
        state["deck_size"] = 6
        state["discard"] = self.initialise_discard()
        return state

    def choose_card(self) -> str:
        if self.strategy == "random":
            return self.random_strategy()
        else:
            raise NotImplementedError("The player does not know the suggested strategy")

    def update_specific_state(self, update_dict: dict, state: dict) -> None:
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

    def run_assigned_actions(self, update_dict: dict):
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

    @staticmethod
    def build_payoff_matrices(left_hand: Iterable, right_hand: Iterable, payoff_lookup):
        """Assumes a priori no A=-B symmetry of payoffs"""
        height = len(left_hand)
        width = len(right_hand)
        left_matrix = np.zeros((height, width))
        right_matrix = np.zeros((height, width))

        for row_index, left_card in enumerate(left_hand):
            for col_index, right_card in enumerate(right_hand):
                left_matrix[row_index, col_index] = payoff_lookup[
                    (left_card, right_card)
                ]["left"]
                right_matrix[row_index, col_index] = payoff_lookup[
                    (left_card, right_card)
                ]["right"]

        return left_matrix, right_matrix

    @staticmethod
    def calculate_nash_equilibrium(left_payoff, right_payoff):

        # We have to round the matrices to integer values
        left_payoff = (left_payoff * 1000).astype("int")
        left_payoff = left_payoff.astype(dtype=pygambit.Rational)
        right_payoff = (right_payoff * 1000).astype("int")
        right_payoff = right_payoff.astype(dtype=pygambit.Rational)

        left_hand_size = left_payoff.shape[0]
        right_hand_size = left_payoff.shape[1]

        game = pygambit.Game.from_arrays(left_payoff, -right_payoff)
        solver = pygambit.nash.ExternalLCPSolver()

        result = solver.solve(game)[0]

        left_vector = np.zeros(left_hand_size)
        right_vector = np.zeros(right_hand_size)

        for i in range(left_hand_size):
            left_vector[i] = result[i]
        for i, j in enumerate(
            range(left_hand_size, (left_hand_size + right_hand_size))
        ):
            right_vector[i] = result[j]

        left_vector = left_vector / left_vector.sum()
        right_vector = right_vector / right_vector.sum()

        return left_vector, right_vector, 1

    @staticmethod
    def simulate_best_strategy(
        own_hand, own_state, other_state, payoff_lookup, num_simulations=5
    ):
        own_hand_size = own_state["hand_size"]
        own_discard = own_state["discard"]
        other_hand_size = other_state["hand_size"]
        other_discard = other_state["discard"]
        other_hand_simulation = RPSPlayer.sample_hand(
            hand_size=other_hand_size,
            discard=other_discard,
            num_samples=num_simulations,
        )
        own_hand_simulation = RPSPlayer.sample_hand(
            hand_size=own_hand_size, discard=own_discard, num_samples=num_simulations
        )

        nash_vectors = []

        for i in range(num_simulations):
            left_payoff, _ = RPSPlayer.build_payoff_matrices(
                left_hand=own_hand,
                right_hand=other_hand_simulation[i, :],
                payoff_lookup=payoff_lookup,
            )
            for j in range(num_simulations):
                _, right_payoff = RPSPlayer.build_payoff_matrices(
                    left_hand=own_hand_simulation[j, :],
                    right_hand=other_hand_simulation[i, :],
                    payoff_lookup=payoff_lookup,
                )
                nash_vector, _, num_results = RPSPlayer.calculate_nash_equilibrium(
                    left_payoff, right_payoff
                )
                if num_results == 1:
                    nash_vectors.append(nash_vector)

        average_strategy = np.mean(nash_vectors, axis=0)
        return average_strategy
