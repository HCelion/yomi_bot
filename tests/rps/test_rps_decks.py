import unittest
from yomi_bot.rps.rps_deck import RPSDeck
import random

class TestRPSDeck(unittest.TestCase):
    
    def setUp(self):
        self.deck = RPSDeck()
        
    def test_deck_length(self):
        deck = RPSDeck()
        assert len(deck) == 9
        
    def test_deck_can_be_shuffled(self):
        deck = RPSDeck()
        assert deck[0] == 'R1'
        random.seed(10)
        deck.shuffle()
        assert len(deck) == 9
        assert deck[0] != 'R1'
        
    def test_deck_shuffled(self):
        deck = RPSDeck()
        assert deck[0] == 'R1'
        random.seed(10)
        shuffled_deck = deck.shuffled()
        shuffled_deck[0] != 'R1'
    
    def test_deck_can_be_drawn_from_while_preserving_order(self):
        deck = RPSDeck()
        assert deck[-3] == 'S1'
        assert deck[-2] == 'S2'
        assert deck[-1] == 'S3' 
        
        draw_cards = deck.draw(1)
        len(draw_cards) == 1
        assert draw_cards[0] == 'S3'
        
        more_cards_drawn = deck.draw(2)
        len(more_cards_drawn) == 2
        assert more_cards_drawn[0] == 'S2'
        assert more_cards_drawn[1] == 'S1'
        
        assert deck[-1] == 'P3'
