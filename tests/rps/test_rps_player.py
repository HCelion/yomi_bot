from yomi_bot.rps.rps_player import RPSPlayer
import random
import unittest

rps_cards = ['R1', 'R2', 'R3', 'S1', 'S2', 'S3', 'P1', 'P2', 'P3']

class TestRPSPlayer(unittest.TestCase):
    
    def test_init_handsize(self):
        player = RPSPlayer()
        assert len(player.hand) == 3
        
        player = RPSPlayer(5)
        assert len(player.hand) == 5
    
    def test_initial_deck_is_shuffled(self):
        random.seed(10)
        player = RPSPlayer()
        assert 'S3' not in player.hand
        
    def test_hand_contents_init(self):
        player = RPSPlayer()
        for card in player.hand:
            assert card in rps_cards
            
    def test_discard_init(self):
        player = RPSPlayer()
        for card in rps_cards:
            player.own_state['discard'][card] = 0
            player.other_state['discard'][card] = 0
            
    def test_scores_are_initialised(self):
        player = RPSPlayer()
        assert player.own_state['score'] == 0
        assert player.other_state['score'] == 0
    
    def test_strategy_is_initialised(self):
        player = RPSPlayer()
        assert player.strategy == 'best_play'
        
        player = RPSPlayer(strategy='random')
        assert player.strategy == 'random'
        
    def test_random_playing(self):
        random.seed(10)
        player = RPSPlayer(strategy='random')
        random.seed(10)
        assert  len(player.hand) == 3
        card = player.choose_card()
        assert card == 'P1'
        assert len(player.hand) == 2

        random.seed(10)
        player = RPSPlayer(strategy='random')
        assert  len(player.hand) == 3
        random.seed(11)
        card = player.choose_card()
        assert card == 'S1'
        assert len(player.hand) == 2
