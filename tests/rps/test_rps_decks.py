import unittest
from yomi_bot.rps.rps_deck import RPSDeck
import random
from collections import Counter


class TestRPSDeck(unittest.TestCase):
    def setUp(self):
        self.deck = RPSDeck()

    def test_deck_length(self):
        deck = RPSDeck()
        assert len(deck) == 9

    def test_deck_can_be_shuffled(self):
        deck = RPSDeck()
        assert deck[0] == "R1"
        random.seed(10)
        deck.shuffle()
        assert len(deck) == 9
        assert deck[0] != "R1"

    def test_deck_shuffled(self):
        deck = RPSDeck()
        assert deck[0] == "R1"
        random.seed(10)
        shuffled_deck = deck.shuffled()
        shuffled_deck[0] != "R1"

    def test_deck_can_be_drawn_from_while_preserving_order(self):
        deck = RPSDeck()
        assert deck[-3] == "S1"
        assert deck[-2] == "S2"
        assert deck[-1] == "S3"

        draw_cards = deck.draw(1)
        len(draw_cards) == 1
        assert draw_cards[0] == "S3"

        more_cards_drawn = deck.draw(2)
        len(more_cards_drawn) == 2
        assert more_cards_drawn[0] == "S2"
        assert more_cards_drawn[1] == "S1"

        assert deck[-1] == "P3"

    def test_too_many_draws_just_empties_deck(self):
        deck = RPSDeck()
        assert len(deck) == 9
        cards = deck.draw(100)
        assert len(cards) == 9
        assert len(deck) == 0

    def test_drawing_from_empty_deck_leaves_empty_hand(self):
        deck = RPSDeck()
        assert len(deck) == 9
        cards = deck.draw(100)
        assert len(deck) == 0
        hand = deck.draw(3)
        assert len(hand) == len(hand.cards) == 0

    def test_adding_decks(self):
        deck1 = RPSDeck()
        deck2 = RPSDeck()
        combined_deck = deck1 + deck2
        assert len(combined_deck) == 18
        counter = Counter(combined_deck.cards)
        for card in counter:
            counter[card] == 2
