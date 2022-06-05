from src.generic_code.arena import Arena
from src.rps.rps_player import RPSPlayer


class RPSArena(Arena):

    player_generator = RPSPlayer

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
        left_suit = left_card[0]
        right_suit = right_card[0]
        left_rank = int(left_card[1])
        right_rank = int(right_card[1])

        if left_suit != right_suit:
            winner = self.determine_suit_winner(left_suit, right_suit)
        elif left_rank > right_rank:
            winner = "left"
        elif left_rank < right_rank:
            winner = "right"
        else:
            winner = "draw"

        left_score, right_score = self.determine_score(
            winner=winner, left_rank=left_rank, right_rank=right_rank
        )

        outcome = self.build_outcome_container(
            winner=winner,
            left_score=left_score,
            right_score=right_score,
            left_card=left_card,
            right_card=right_card,
        )

        return outcome

    def determine_suit_winner(self, left_suit, right_suit):
        # Assumes that it is called with different suites, responsibility lies
        # with the caller

        if left_suit == "R":
            # Rock beats Scissors
            if right_suit == "S":
                return "left"
            # Rock loses to paper
            elif right_suit == "P":
                return "right"
        elif left_suit == "P":
            # Paper covers rock
            if right_suit == "R":
                return "left"
            # Paper gets cut by scissors
            elif right_suit == "S":
                return "right"
        elif left_suit == "S":
            # Sciccor gets crushed by rock
            if right_suit == "R":
                return "right"
            # Paper cuts scissor
            elif right_suit == "P":
                return "left"

    def build_outcome_container(
        self, winner, left_score, right_score, left_card, right_card
    ):

        outcome = {
            "left": {"state": {"score": left_score, "discards": [left_card]}},
            "right": {"state": {"score": right_score, "discards": [right_card]}},
            "winner": winner,
        }
        # Both teams always draw 1
        outcome["left"]["actions"] = {"draw": 1}
        outcome["right"]["actions"] = {"draw": 1}
        return outcome

    def determine_score(self, winner, left_rank, right_rank):

        if winner == "left":
            left_score = 4 - left_rank
            right_score = 0
        elif winner == "right":
            left_score = 0
            right_score = 4 - right_rank
        else:  # Draw
            left_score = 0
            right_score = 0

        return left_score, right_score
