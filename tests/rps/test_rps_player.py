from yomi_bot.rps.rps_player import RPSPlayer
import random
import unittest
from copy import deepcopy

rps_cards = ['R1', 'R2', 'R3', 'S1', 'S2', 'S3', 'P1', 'P2', 'P3']

class TestRPSPlayerInit(unittest.TestCase):
    
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
        
    def test_side_is_initialised(self):
        player = RPSPlayer()
        assert player.side == 'left'
        assert player.other_side == 'right'
        
        player = RPSPlayer(side='right')
        assert player.side == 'right'
        assert player.other_side == 'left'


class TestPlaying(unittest.TestCase):

        
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

class TestUpdating(unittest.TestCase):
    
    def setUp(self):
        self.update_dict = {
        'left':{
            'state': {'score':1, 'discards':['R1', 'R2']},
            'actions':{'draw':2}
        }, 
        'right':{
            'state': {'score':-1, 'discards':['S1', 'S2']},
            'actions':{'draw':1}
        }}
    
    def test_own_state_updates_left(self):
        player = RPSPlayer(side='left')
        assert player.own_state['score'] == 0
        assert player.own_state['discard']['R1'] == 0
        assert player.own_state['discard']['R2'] == 0
        
        player.update_state(self.update_dict)
        assert player.own_state['score'] == 1
        assert player.own_state['discard']['R1'] == 1
        assert player.own_state['discard']['R2'] == 1
        
    def test_own_state_updates_right(self):
        player = RPSPlayer(side='right')
        assert player.own_state['score'] == 0
        assert player.own_state['discard']['S1'] == 0
        assert player.own_state['discard']['S2'] == 0
        
        player.update_state(self.update_dict)
        assert player.own_state['score'] == -1
        assert player.own_state['discard']['S1'] == 1
        assert player.own_state['discard']['S2'] == 1
    
    def test_other_state_update_left(self):
        player = RPSPlayer(side='left')
        assert player.other_state['score'] == 0
        assert player.other_state['discard']['S1'] == 0
        assert player.other_state['discard']['S2'] == 0        
        
        player.update_state(self.update_dict)
        assert player.other_state['score'] == - 1
        assert player.other_state['discard']['S1'] == 1
        assert player.other_state['discard']['S2'] == 1
        
    def test_other_state_update_right(self):
        player = RPSPlayer(side='right')
        assert player.other_state['score'] == 0
        assert player.other_state['discard']['R1'] == 0
        assert player.other_state['discard']['R2'] == 0        
        
        player.update_state(self.update_dict)
        assert player.other_state['score'] == 1
        assert player.other_state['discard']['R1'] == 1
        assert player.other_state['discard']['R2'] == 1
        
    def test_number_drawn_correct_left(self):
        player = RPSPlayer(side='left')
        original_hand = deepcopy(player.hand.cards)
        assert len(original_hand) == 3
        
        player.update_state(self.update_dict)
        
        assert len(player.hand) == 5
        
        # Test that all original cards are still in hand after drawing
        for card in original_hand:
            assert card in player.hand

    def test_number_drawn_correct_right(self):
        player = RPSPlayer(side='right')
        original_hand = deepcopy(player.hand.cards)
        assert len(original_hand) == 3
        
        player.update_state(self.update_dict)
        
        assert len(player.hand) == 4
        
        # Test that all original cards are still in hand after drawing
        for card in original_hand:
            assert card in player.hand        
