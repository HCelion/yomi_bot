import unittest
from yomi_bot.rps.rps_arena import RPSArena
import random


class MockPlayerGenerator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.name = "mock_generator"
        self.hand = [1, 2, 3]
        self.deck = [4, 5, 6]
        self.own_state = {"score": 0}


class TestRPSArenaInit(unittest.TestCase):
    def test_init_hand_sizes(self):
        arena = RPSArena()
        assert len(arena.left_player.hand) == 3
        assert len(arena.right_player.hand) == 3

    def test_sides_aligned_correctly(self):
        arena = RPSArena()
        assert arena.left_player.side == "left"
        assert arena.right_player.side == "right"

    def test_strategies_pass(self):
        arena = RPSArena(
            left_strategy="dumb_strategy", right_strategy="even_dumber_strategy"
        )
        assert arena.left_player.strategy == "dumb_strategy"
        assert arena.right_player.strategy == "even_dumber_strategy"

    def test_injecting_generators(self):
        arena = RPSArena(left_generator=MockPlayerGenerator)
        assert arena.left_player.name == "mock_generator"
        assert not hasattr(arena.right_player, "name")

        arena = RPSArena(right_generator=MockPlayerGenerator)
        assert arena.right_player.name == "mock_generator"
        assert not hasattr(arena.left_player, "name")


class TestRPSEvaluations(unittest.TestCase):
    def setUp(self):
        self.arena = RPSArena()

    def test_is_game_not_ended_when_both_handsn_non_empty(self):
        arena = RPSArena()
        left_player = MockPlayerGenerator()
        right_player = MockPlayerGenerator()
        assert not arena.is_game_end(left_player, right_player)

    def test_either_player_empty_sufficient(self):
        arena = RPSArena()
        left_player = MockPlayerGenerator()
        right_player = MockPlayerGenerator()
        left_player.hand = []
        assert arena.is_game_end(left_player, right_player)
        assert arena.is_game_end(right_player, left_player)

    def test_both_empty(self):
        arena = RPSArena()
        left_player = MockPlayerGenerator()
        left_player.hand = []
        assert arena.is_game_end(left_player, left_player)

    def test_game_outcome_left_wins(self):
        arena = RPSArena()
        left_player = MockPlayerGenerator()
        right_player = MockPlayerGenerator()
        left_player.own_state["score"] = 5
        right_player.own_state["score"] = 2
        game_outcome = arena.evaluate_game_outcome(left_player, right_player)
        assert game_outcome["winner"] == "left"
        assert game_outcome["left_score"] == 5
        assert game_outcome["right_score"] == 2

    def test_game_outcome_right_wins(self):
        arena = RPSArena()
        left_player = MockPlayerGenerator()
        right_player = MockPlayerGenerator()
        left_player.own_state["score"] = 3
        right_player.own_state["score"] = 18
        game_outcome = arena.evaluate_game_outcome(left_player, right_player)
        assert game_outcome["winner"] == "right"
        assert game_outcome["left_score"] == 3
        assert game_outcome["right_score"] == 18

    def test_draw(self):
        arena = RPSArena()
        left_player = MockPlayerGenerator()
        right_player = MockPlayerGenerator()
        left_player.own_state["score"] = 12
        right_player.own_state["score"] = 12
        game_outcome = arena.evaluate_game_outcome(left_player, right_player)
        assert game_outcome["winner"] == "draw"
        assert game_outcome["left_score"] == 12
        assert game_outcome["right_score"] == 12

    def test_build_outcome_container(self):
        outcome = self.arena.build_outcome_container(
            winner="whatever",
            left_score=1,
            right_score=2,
            left_card="R1",
            right_card="S2",
        )
        assert outcome["winner"] == "whatever"
        assert outcome["left"]["state"]["score"] == 1
        assert outcome["right"]["state"]["score"] == 2
        assert outcome["left"]["actions"]["draw"] == 1
        assert len(outcome["left"]["state"]["discards"]) == 1
        assert outcome["left"]["state"]["discards"][0] == "R1"
        assert outcome["right"]["actions"]["draw"] == 1
        assert len(outcome["right"]["state"]["discards"]) == 1
        assert outcome["right"]["state"]["discards"][0] == "S2"

    def test_determine_score(self):

        left_score, right_score = self.arena.determine_score("left", 3, 1)
        assert left_score == 1
        assert right_score == 0

        left_score, right_score = self.arena.determine_score("left", 2, 1)
        assert left_score == 2
        assert right_score == 0

        left_score, right_score = self.arena.determine_score("left", 3, 1)
        assert left_score == 1
        assert right_score == 0

        left_score, right_score = self.arena.determine_score("right", 3, 1)
        assert left_score == 0
        assert right_score == 3

        left_score, right_score = self.arena.determine_score("right", 3, 2)
        assert left_score == 0
        assert right_score == 2

        left_score, right_score = self.arena.determine_score("right", 3, 3)
        assert left_score == 0
        assert right_score == 1

        left_score, right_score = self.arena.determine_score("draw", 5, 5)
        assert left_score == 0
        assert right_score == 0

    def test_suite_winners(self):
        winner = self.arena.determine_suit_winner("R", "P")
        assert winner == "right"
        winner = self.arena.determine_suit_winner("R", "S")
        assert winner == "left"
        winner = self.arena.determine_suit_winner("P", "R")
        assert winner == "left"
        winner = self.arena.determine_suit_winner("P", "S")
        assert winner == "right"
        winner = self.arena.determine_suit_winner("S", "R")
        assert winner == "right"
        winner = self.arena.determine_suit_winner("S", "P")
        assert winner == "left"

    def test_one_round_win_by_rank(self):
        outcome = self.arena.evaluate_round_outcome("R1", "R3")
        assert outcome["winner"] == "right"
        assert outcome["left"]["state"]["score"] == 0
        assert outcome["right"]["state"]["score"] == 1

        outcome = self.arena.evaluate_round_outcome("R2", "R1")
        assert outcome["winner"] == "left"
        assert outcome["left"]["state"]["score"] == 2
        assert outcome["right"]["state"]["score"] == 0

    def test_one_round_won_by_suit(self):
        outcome = self.arena.evaluate_round_outcome("R1", "P3")
        assert outcome["winner"] == "right"
        assert outcome["left"]["state"]["score"] == 0
        assert outcome["right"]["state"]["score"] == 1

        outcome = self.arena.evaluate_round_outcome("S2", "P3")
        assert outcome["winner"] == "left"
        assert outcome["left"]["state"]["score"] == 2
        assert outcome["right"]["state"]["score"] == 0


class TestFullGame(unittest.TestCase):
    def test_full_run_has_empty_hands(self):
        all_winners = []
        score_diffs = []

        random.seed(13)
        for _ in range(1000):
            arena = RPSArena(left_strategy="random", right_strategy="random")
            outcome = arena.play_game()
            all_winners.append(outcome["winner"])
            score_diffs.append(outcome["left_score"] - outcome["right_score"])
            assert len(arena.left_player.hand) == 0
            assert len(arena.left_player.deck) == 0
            assert len(arena.right_player.hand) == 0
            assert len(arena.right_player.deck) == 0

        assert "left" in all_winners
        assert "right" in all_winners
        assert "draw" in all_winners
        assert max(score_diffs) == 18  # The perfect game for left
        assert min(score_diffs) == -18  # The perfect game for right

    def test_discards_are_full(self):
        arena = RPSArena(left_strategy="random", right_strategy="random")
        _ = arena.play_game()
        for card in arena.left_player.own_state["discard"]:
            # All players should agree that their and the opposing player's
            # discard are full
            arena.left_player.own_state["discard"][card] == 1
            arena.left_player.other_state["discard"][card] == 1
            arena.right_player.own_state["discard"][card] == 1
            arena.right_player.other_state["discard"][card] == 1

        assert arena.left_player.own_state["hand_size"] == 0
        assert arena.left_player.own_state["deck_size"] == 0
        assert arena.left_player.other_state["hand_size"] == 0
        assert arena.left_player.other_state["deck_size"] == 0

        assert arena.right_player.own_state["hand_size"] == 0
        assert arena.right_player.own_state["deck_size"] == 0
        assert arena.right_player.other_state["hand_size"] == 0
        assert arena.right_player.other_state["deck_size"] == 0
