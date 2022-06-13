from src.rps.rps_player import RPSPlayer
from src.rps.rps_deck import rps_cards
import src.rps.rps_rules as rps
import random
import numpy as np
from copy import deepcopy
import unittest
import warnings
from collections import Counter

warnings.simplefilter("ignore")
np.seterr(all="ignore")


class TestRPSPlayerInit(unittest.TestCase):
    def test_init_handsize(self):
        player = RPSPlayer()
        assert len(player.hand) == 3

        player = RPSPlayer(5)
        assert len(player.hand) == 5

    def test_initial_deck_is_shuffled(self):
        random.seed(10)
        player = RPSPlayer()
        assert "S3" not in player.hand

    def test_hand_contents_init(self):
        player = RPSPlayer()
        for card in player.hand:
            assert card in rps_cards

    def test_discard_init(self):
        player = RPSPlayer()
        for card in rps_cards:
            player.own_state["discard"][card] = 0
            player.other_state["discard"][card] = 0

    def test_scores_are_initialised(self):
        player = RPSPlayer()
        assert player.own_state["score"] == 0
        assert player.other_state["score"] == 0

    def test_strategy_is_initialised(self):
        player = RPSPlayer()
        assert player.strategy == "best_play"

        player = RPSPlayer(strategy="random")
        assert player.strategy == "random"

    def test_side_is_initialised(self):
        player = RPSPlayer()
        assert player.side == "left"
        assert player.other_side == "right"

        player = RPSPlayer(side="right")
        assert player.side == "right"
        assert player.other_side == "left"

    def test_hand_size_is_initialised(self):
        player = RPSPlayer()
        assert player.own_state["hand_size"] == 3
        assert player.other_state["hand_size"] == 3

    def test_deck_size_is_initialised(self):
        player = RPSPlayer()
        assert player.own_state["deck_size"] == 6
        assert player.other_state["deck_size"] == 6

    def test_num_simulations(self):
        player = RPSPlayer()
        assert player.num_simulations == 10

        player2 = RPSPlayer(num_simulations=100)
        assert player2.num_simulations == 100


class TestPlaying(unittest.TestCase):
    def test_random_playing(self):
        random.seed(10)
        player = RPSPlayer(strategy="random")
        random.seed(10)
        assert len(player.hand) == 3
        card = player.choose_card()
        assert card == "P1"
        assert len(player.hand) == 2

        random.seed(10)
        player = RPSPlayer(strategy="random")
        assert len(player.hand) == 3
        random.seed(11)
        card = player.choose_card()
        assert card == "S1"
        assert len(player.hand) == 2


class TestUpdating(unittest.TestCase):
    def setUp(self):
        self.update_dict = {
            "left": {
                "state": {"score": 1, "discards": ["R1", "R2"]},
                "actions": {"draw": 2},
            },
            "right": {
                "state": {"score": -1, "discards": ["S1", "S2"]},
                "actions": {"draw": 1},
            },
        }

    def test_own_state_updates_left(self):
        player = RPSPlayer(side="left")
        assert player.own_state["score"] == 0
        assert player.own_state["discard"]["R1"] == 0
        assert player.own_state["discard"]["R2"] == 0

        player.update_state(self.update_dict)
        assert player.own_state["score"] == 1
        assert player.own_state["discard"]["R1"] == 1
        assert player.own_state["discard"]["R2"] == 1

    def test_own_state_updates_right(self):
        player = RPSPlayer(side="right")
        assert player.own_state["score"] == 0
        assert player.own_state["discard"]["S1"] == 0
        assert player.own_state["discard"]["S2"] == 0

        player.update_state(self.update_dict)
        assert player.own_state["score"] == -1
        assert player.own_state["discard"]["S1"] == 1
        assert player.own_state["discard"]["S2"] == 1

    def test_other_state_update_left(self):
        player = RPSPlayer(side="left")
        assert player.other_state["score"] == 0
        assert player.other_state["discard"]["S1"] == 0
        assert player.other_state["discard"]["S2"] == 0

        player.update_state(self.update_dict)
        assert player.other_state["score"] == -1
        assert player.other_state["discard"]["S1"] == 1
        assert player.other_state["discard"]["S2"] == 1

    def test_other_state_update_right(self):
        player = RPSPlayer(side="right")
        assert player.other_state["score"] == 0
        assert player.other_state["discard"]["R1"] == 0
        assert player.other_state["discard"]["R2"] == 0

        player.update_state(self.update_dict)
        assert player.other_state["score"] == 1
        assert player.other_state["discard"]["R1"] == 1
        assert player.other_state["discard"]["R2"] == 1

    def test_number_drawn_correct_left(self):
        player = RPSPlayer(side="left")
        original_hand = deepcopy(player.hand.cards)
        assert len(original_hand) == 3

        player.update_state(self.update_dict)

        assert len(player.hand) == 5

        # Test that all original cards are still in hand after drawing
        for card in original_hand:
            assert card in player.hand

    def test_number_drawn_correct_right(self):
        player = RPSPlayer(side="right")
        original_hand = deepcopy(player.hand.cards)
        assert len(original_hand) == 3

        player.update_state(self.update_dict)

        assert len(player.hand) == 4

        # Test that all original cards are still in hand after drawing
        for card in original_hand:
            assert card in player.hand

    def test_hand_deck_sizes_update_correctly_left(self):
        player = RPSPlayer(side="left")
        assert player.own_state["hand_size"] == 3
        assert player.own_state["deck_size"] == 6
        assert player.other_state["hand_size"] == 3
        assert player.other_state["deck_size"] == 6

        player.update_state(self.update_dict)

        assert player.own_state["hand_size"] == 3  # Discard 2, draw 2
        assert player.own_state["deck_size"] == 4
        assert player.other_state["hand_size"] == 2  # Discard 2, draw 1
        assert player.other_state["deck_size"] == 5

    def test_hand_deck_sizes_update_correctly_right(self):
        player = RPSPlayer(side="right")
        assert player.own_state["hand_size"] == 3
        assert player.own_state["deck_size"] == 6
        assert player.other_state["hand_size"] == 3
        assert player.other_state["deck_size"] == 6

        player.update_state(self.update_dict)

        assert player.own_state["hand_size"] == 2
        assert player.own_state["deck_size"] == 5
        assert player.other_state["hand_size"] == 3
        assert player.other_state["deck_size"] == 4

    def test_overdrawing(self):
        player = RPSPlayer(side="left")
        player.own_state["deck_size"] = 1
        assert player.own_state["hand_size"] == 3

        # Player draws 2, but has only 1 in deck, and discards 2
        player.update_state(self.update_dict)
        assert player.own_state["deck_size"] == 0
        assert player.own_state["hand_size"] == 2


class TestHandSampling(unittest.TestCase):
    def setUp(self):
        self.empty_discard = {card: 0 for card in rps_cards}

    def test_sizes_are_correct(self):
        sample_array = RPSPlayer.sample_hand(
            hand_size=3, discard=self.empty_discard, num_samples=50
        )
        assert sample_array.shape == (50, 3)

    def test_cards_are_contained(self):
        np.random.seed(10)
        sample_array = RPSPlayer.sample_hand(
            hand_size=3, discard=self.empty_discard, num_samples=100
        )
        for card in rps_cards:
            assert card in sample_array

    def test_items_excluded(self):

        discard = deepcopy(self.empty_discard)
        discard["S1"] = 1
        sample_array = RPSPlayer.sample_hand(
            hand_size=3, discard=discard, num_samples=100
        )

        assert "S1" not in sample_array


class TestBestPlay(unittest.TestCase):
    def setUp(self):
        self.payoff_lookup = rps.generate_rps_payoff_lookup()
        self.empty_discard = {card: 0 for card in rps_cards}

    def test_payoff_matrix_is_correct(self):
        left_hand = ["S1", "S2", "S3"]
        right_hand = ["S2", "P1"]

        left_payoff, right_payoff = RPSPlayer.build_payoff_matrices(
            left_hand, right_hand, self.payoff_lookup
        )

        assert left_payoff.shape == right_payoff.shape == (3, 2)

        ideal_left_result = np.array([[-2, +3], [0, +2], [+1, +1]])

        self.assertEqual(np.abs(left_payoff - ideal_left_result).sum(), 0)
        # right matrix shoudl be just the negative of the left matrix
        self.assertEqual(np.abs(right_payoff + ideal_left_result).sum(), 0)

    def test_nash_vector_extraction(self):
        left_hand = ["S1", "S2", "S3"]
        right_hand = ["S2", "P1"]

        left_payoff, right_payoff = RPSPlayer.build_payoff_matrices(
            left_hand, right_hand, self.payoff_lookup
        )

        left_balance, right_balance = RPSPlayer.calculate_nash_equilibrium(
            left_payoff, right_payoff
        )
        assert len(left_balance) == 3
        assert len(right_balance) == 2

        assert left_balance.sum() == 1
        assert right_balance.sum() == 1

        # S3 is the doiminant strategy for left, S2 for right
        assert np.sum(left_balance - np.array([0, 0, 1])) == 0
        assert np.sum(right_balance - np.array([1, 0])) == 0

    def test_attack_weight_calculation(self):
        own_discard = deepcopy(self.empty_discard)
        other_discard = deepcopy(self.empty_discard)
        other_discard["S2"] = 1
        other_hand_simulation = RPSPlayer.sample_hand(
            hand_size=2,
            discard=other_discard,
            num_samples=5,
        )
        own_hand_simulation = RPSPlayer.sample_hand(
            hand_size=3, discard=own_discard, num_samples=10
        )

        attack_weights = RPSPlayer.generate_average_attack_vector(
            own_simulations=own_hand_simulation,
            other_simulations=other_hand_simulation,
            payoff_lookup=self.payoff_lookup,
        )

        # 'S2' is already in discard
        assert "S2" not in attack_weights
        total_weight = 0
        for _, weight in attack_weights.items():
            assert weight >= 0
            total_weight += weight
        self.assertAlmostEqual(total_weight, 1, places=5)

    def test_extract_thompson_frequencies(self):
        own_hand = ["S1", "S2"]
        attack_weights = {"S3": 0.5, "P1": 0.5}

        thompson_weights = RPSPlayer.extract_thompson_probs(
            own_hand=own_hand,
            attack_weights=attack_weights,
            payoff_lookup=self.payoff_lookup,
        )

        assert len(thompson_weights) == 2
        # Wins against paper with more utility
        assert thompson_weights["S1"] == 0.5 * 0.5 + 0.5
        assert thompson_weights["S2"] == 1 - thompson_weights["S1"]

        own_hand = ["S1", "P1", "R1"]
        attack_weights = {"S3": 1 / 3, "P3": 1 / 3, "R3": 1 / 3}

        thompson_weights = RPSPlayer.extract_thompson_probs(
            own_hand=own_hand,
            attack_weights=attack_weights,
            payoff_lookup=self.payoff_lookup,
        )

        assert len(thompson_weights) == 3
        # Wins against paper with more utility
        assert thompson_weights["S1"] == 1 / 3
        assert thompson_weights["P1"] == 1 / 3
        assert thompson_weights["R1"] == 1 / 3

    def test_simulate_best_strategy(self):
        own_hand = ["S1", "S2"]
        own_state = {"hand_size": 2, "discard": self.empty_discard}
        other_state = {"hand_size": 1, "discard": self.empty_discard}
        strategy = RPSPlayer.simulate_best_strategy(
            own_hand=own_hand,
            own_state=own_state,
            other_state=other_state,
            payoff_lookup=self.payoff_lookup,
            num_simulations=5,
        )

        assert len(strategy) == 2

        weights = 0.0
        for _, weight in strategy.items():
            weights += weight
        self.assertAlmostEqual(weights, 1, places=5)

    def test_sampling_from_dict(self):
        example_dict = {"one": 0.5, "two": 0.25, "three": 0.25, "four": 0.0}
        all_samples = []
        for _ in range(10000):
            all_samples.append(RPSPlayer.sample_from_dict(example_dict))

        counter = Counter(all_samples)

        assert "four" not in counter
        assert counter["one"] > counter["two"] * 1.9
        assert counter["one"] < counter["two"] * 2.1
        assert counter["two"] > 0.9 * counter["three"]
        assert counter["two"] < 1.1 * counter["three"]

    def test_best_strategy_runs(self):
        player = RPSPlayer(num_simulations=2, strategy="best")
        assert len(player.hand) == 3

        hand_copy = deepcopy(player.hand)
        card = player.choose_card()

        assert card in hand_copy
        assert len(player.hand) == 2
        assert card not in player.hand
