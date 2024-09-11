from src.rps.rps_deck import rps_cards


def determine_suit_winner(left_suit, right_suit):
    # Assumes that it is called with different suites, responsibility lies
    # with the caller

    if left_suit == "T":
        # Rock beats Scissors
        if right_suit == "D":
            return "left"
        # Rock loses to paper
        elif right_suit == "A":
            return "right"
    elif left_suit == "A":
        # Paper covers rock
        if right_suit == "T":
            return "left"
        # Paper gets cut by scissors
        elif right_suit == "D":
            return "right"
    elif left_suit == "D":
        # Sciccor gets crushed by rock
        if right_suit == "T":
            return "right"
        # Paper cuts scissor
        elif right_suit == "A":
            return "left"


def determine_score(winner, left_rank, right_rank):

    if winner == "left":
        left_score = left_rank
        right_score = 0
    elif winner == "right":
        left_score = 0
        right_score = right_rank
    else:  # Draw
        left_score = 0
        right_score = 0

    return left_score, right_score


def determine_rps_outcome(left_card, right_card):

    left_speed = left_card.speed
    right_speed = right_card.speed

    if left_suit != right_suit:
        winner = determine_suit_winner(left_card.type, right_card.type)

    elif left_speed > right_speed:
        winner = "left"
    elif left_speed < right_speed:
        winner = "right"
    else:
        winner = "draw"

    left_score, right_score = determine_score(
        winner=winner, left_rank=left_card.damage, right_rank=right_card.damage
    )

    return winner, left_score, right_score


def generate_rps_payoff_lookup():

    pay_off_matrix = {}
    for left_card in rps_cards:
        for right_card in rps_cards:
            winner, left_score, right_score = determine_rps_outcome(left_card, right_card)
            score_diff = left_score - right_score
            pay_off_matrix[(left_card, right_card)] = {
                "left": score_diff,
                "right": -score_diff,
                "winner": winner,
                "left_score": left_score,
                "right_score": right_score,
            }

    return pay_off_matrix
