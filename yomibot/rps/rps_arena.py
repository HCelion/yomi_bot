from yomibot.generic_code.arena import Arena
from yomibot.rps.rps_player import RPSPlayer
import yomibot.rps.rps_rules as rps


class RPSArena(Arena):

    player_generator = RPSPlayer
    payoff_lookup = rps.generate_rps_payoff_lookup()

    def is_game_end(self, left_player, right_player):
        return (len(left_player.hand) == 0) | (len(right_player.hand) == 0)

    def evaluate_game_outcome(self, left_player, right_player):
        left_score = left_player.own_state["score"]
        right_score = right_player.own_state["score"]

        if left_score > right_score:
            winner = "left"
        elif right_score > left_score:
            winner = "right"
        else:
            winner = "draw"

        return {"winner": winner, "left_score": left_score, "right_score": right_score}

    def evaluate_round_outcome(self, left_card, right_card):
        outcome = self.payoff_lookup[(left_card, right_card)]

        outcome = self.build_outcome_container(
            winner=outcome["winner"],
            left_score=outcome["left_score"],
            right_score=outcome["right_score"],
            left_card=left_card,
            right_card=right_card,
        )

        return outcome

    @staticmethod
    def build_outcome_container(winner, left_score, right_score, left_card, right_card):

        outcome = {
            "left": {"state": {"score": left_score, "discards": [left_card]}},
            "right": {"state": {"score": right_score, "discards": [right_card]}},
            "winner": winner,
        }
        # Both teams always draw 1
        outcome["left"]["actions"] = {"draw": 1}
        outcome["right"]["actions"] = {"draw": 1}
        return outcome
