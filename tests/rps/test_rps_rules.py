from yomibot.rps.rps_rules import (
    determine_suit_winner,
    determine_score,
    determine_rps_outcome,
    generate_rps_payoff_lookup,
)
import unittest


class TestRules(unittest.TestCase):
    def test_suite_winners(self):
        winner = determine_suit_winner("R", "P")
        assert winner == "right"
        winner = determine_suit_winner("R", "S")
        assert winner == "left"
        winner = determine_suit_winner("P", "R")
        assert winner == "left"
        winner = determine_suit_winner("P", "S")
        assert winner == "right"
        winner = determine_suit_winner("S", "R")
        assert winner == "right"
        winner = determine_suit_winner("S", "P")
        assert winner == "left"

    def test_determine_score(self):

        left_score, right_score = determine_score("left", 3, 1)
        assert left_score == 1
        assert right_score == 0

        left_score, right_score = determine_score("left", 2, 1)
        assert left_score == 2
        assert right_score == 0

        left_score, right_score = determine_score("left", 3, 1)
        assert left_score == 1
        assert right_score == 0

        left_score, right_score = determine_score("right", 3, 1)
        assert left_score == 0
        assert right_score == 3

        left_score, right_score = determine_score("right", 3, 2)
        assert left_score == 0
        assert right_score == 2

        left_score, right_score = determine_score("right", 3, 3)
        assert left_score == 0
        assert right_score == 1

        left_score, right_score = determine_score("draw", 5, 5)
        assert left_score == 0
        assert right_score == 0

    def test_rps_outcome(self):

        winner, left_score, right_score = determine_rps_outcome("R2", "S1")
        assert winner == "left"
        assert left_score == 2
        assert right_score == 0

        winner, left_score, right_score = determine_rps_outcome("R1", "S1")
        assert winner == "left"
        assert left_score == 3
        assert right_score == 0

        winner, left_score, right_score = determine_rps_outcome("P3", "P1")
        assert winner == "left"
        assert left_score == 1
        assert right_score == 0

        winner, left_score, right_score = determine_rps_outcome("P3", "S3")
        assert winner == "right"
        assert left_score == 0
        assert right_score == 1

    def test_payoff_matrix(self):

        payoff_matrix = generate_rps_payoff_lookup()

        outcome_one = payoff_matrix[("R2", "S2")]
        assert outcome_one["winner"] == "left"
        assert outcome_one["left_score"] == 2
        assert outcome_one["right_score"] == 0
        assert outcome_one["left"] == 2
        assert outcome_one["right"] == -2

        outcome_two = payoff_matrix[("P1", "P1")]
        assert outcome_two["winner"] == "draw"
        assert outcome_two["left_score"] == 0
        assert outcome_two["right_score"] == 0
        assert outcome_two["left"] == 0
        assert outcome_two["right"] == 0

        outcome_two = payoff_matrix[("P3", "S1")]
        assert outcome_two["winner"] == "right"
        assert outcome_two["left_score"] == 0
        assert outcome_two["right_score"] == 3
        assert outcome_two["left"] == -3
        assert outcome_two["right"] == 3
