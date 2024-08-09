from yomibot.rps.rps_deck import rps_cards


def determine_suit_winner(left_suit, right_suit):
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


def determine_score(winner, left_rank, right_rank):

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


def determine_rps_outcome(left_card, right_card):

    left_suit = left_card[0]
    right_suit = right_card[0]
    left_rank = int(left_card[1])
    right_rank = int(right_card[1])

    if left_suit != right_suit:
        winner = determine_suit_winner(left_suit, right_suit)
    elif left_rank > right_rank:
        winner = "left"
    elif left_rank < right_rank:
        winner = "right"
    else:
        winner = "draw"

    left_score, right_score = determine_score(
        winner=winner, left_rank=left_rank, right_rank=right_rank
    )

    return winner, left_score, right_score


def generate_rps_payoff_lookup():

    pay_off_matrix = {}
    for left_card in rps_cards:
        for right_card in rps_cards:
            winner, left_score, right_score = determine_rps_outcome(
                left_card, right_card
            )
            score_diff = left_score - right_score
            pay_off_matrix[(left_card, right_card)] = {
                "left": score_diff,
                "right": -score_diff,
                "winner": winner,
                "left_score": left_score,
                "right_score": right_score,
            }

    return pay_off_matrix
