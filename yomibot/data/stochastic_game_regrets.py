import nashpy as nash
import numpy as np


def get_payout_matrix(payout_function):
    A = np.array(
        [
            [payout_function("Even", "Even"), payout_function("Even", "Odd")],
            [payout_function("Odd", "Even"), payout_function("Odd", "Odd")],
        ]
    )
    return A


def get_penny_nash_equilibria(A, return_value=False):
    rps = nash.Game(A, -A)
    eqs = rps.support_enumeration()
    player_1, player_2 = list(eqs)[0]
    order = ("Even", "Odd")
    optimum_1 = {action: prob for action, prob in zip(order, player_1)}
    optimum_2 = {action: prob for action, prob in zip(order, player_2)}

    if return_value:
        p1_value = np.array(player_1).T @ A @ np.array(player_2)
        return optimum_1, optimum_2, p1_value
    return optimum_1, optimum_2


def get_current_value(payout_matrix, player_1_policy, player_2_policy, prob1=1, prob2=1):
    p1_policy_vector = np.array([player_1_policy["Even"], player_1_policy["Odd"]])
    p2_policy_vector = np.array([player_2_policy["Even"], player_2_policy["Odd"]])
    value = p1_policy_vector.T @ payout_matrix @ p2_policy_vector

    even_vector = np.array([1, 0])
    odd_vector = np.array([0, 1])

    regrets1_even = (even_vector @ payout_matrix @ p2_policy_vector) * prob2
    regrets1_odd = (odd_vector @ payout_matrix @ p2_policy_vector) * prob2
    p1_regrets = {"Even": regrets1_even - value, "Odd": regrets1_odd - value}

    regrets2_even = (p1_policy_vector @ (-payout_matrix) @ even_vector) * prob1
    regrets2_odd = (p1_policy_vector @ (-payout_matrix) @ odd_vector) * prob1
    p1_regrets = {"Even": regrets1_even - value, "Odd": regrets1_odd - value}
    p2_regrets = {"Even": regrets2_even + value, "Odd": regrets2_odd + value}

    return value, p1_regrets, p2_regrets


def get_optimal_policy(payout_dictionary, scenario="choice"):
    payout_matrix_2 = get_payout_matrix(payout_dictionary[2][0])
    state_2_optimum1, state_2_optimum2, state_2_value = get_penny_nash_equilibria(
        payout_matrix_2, return_value=True
    )

    payout_matrix_3 = get_payout_matrix(payout_dictionary[3][0])
    state_3_optimum1, state_3_optimum2, state_3_value = get_penny_nash_equilibria(
        payout_matrix_3, return_value=True
    )

    payout_matrix_1_raw = get_payout_matrix(payout_dictionary[1][0])

    if scenario == "choice":
        payout_matrix_1 = payout_matrix_1_raw + np.array(
            [[state_2_value, state_2_value], [state_3_value, state_3_value]]
        )
    elif scenario == "win":
        payout_matrix_1 = payout_matrix_1_raw + np.array(
            [[state_2_value, state_3_value], [state_3_value, state_2_value]]
        )
    else:
        payout_matrix_1 = payout_matrix_1_raw + (0.5) * (state_3_value + state_2_value)

    state_1_optimum1, state_1_optimum2, state_1_value = get_penny_nash_equilibria(
        payout_matrix_1, return_value=True
    )
    optima = {}
    optima[1] = {"alpha": state_1_optimum1["Even"], "beta": state_1_optimum2["Even"]}
    optima[2] = {"alpha": state_2_optimum1["Even"], "beta": state_2_optimum2["Even"]}
    optima[3] = {"alpha": state_3_optimum1["Even"], "beta": state_3_optimum2["Even"]}
    return optima


def get_stochastic_penny(payout_dictionary, model_dictionary, scenario="choice"):
    payout_matrix_2 = get_payout_matrix(payout_dictionary[2][0])
    state_2_optimum1, state_2_optimum2, state_2_opti_value = get_penny_nash_equilibria(
        payout_matrix_2, return_value=True
    )

    game_2_current_value, p1_regrets_2, p2_regrets_2 = get_current_value(
        payout_matrix_2, model_dictionary[0][2], model_dictionary[1][2]
    )

    payout_matrix_3 = get_payout_matrix(payout_dictionary[3][0])
    state_3_optimum1, state_3_optimum2, state_3_value = get_penny_nash_equilibria(
        payout_matrix_3, return_value=True
    )

    game_3_current_value, p1_regrets_3, p2_regrets_3 = get_current_value(
        payout_matrix_3, model_dictionary[0][3], model_dictionary[1][3]
    )

    payout_matrix_1_raw = get_payout_matrix(payout_dictionary[1][0])

    if scenario == "choice":
        payout_matrix_1 = payout_matrix_1_raw + np.array(
            [
                [game_2_current_value, game_2_current_value],
                [game_3_current_value, game_3_current_value],
            ]
        )
    elif scenario == "win":
        payout_matrix_1 = payout_matrix_1_raw + np.array(
            [
                [game_2_current_value, game_3_current_value],
                [game_2_current_value, game_3_current_value],
            ]
        )
    else:
        payout_matrix_1 = payout_matrix_1_raw + (0.5) * (
            game_2_current_value + game_3_current_value
        )

    game_1_current_value, p1_regrets_1, p2_regrets_1 = get_current_value(
        payout_matrix_1, model_dictionary[0][1], model_dictionary[1][1]
    )

    regrets1 = {1: p1_regrets_1, 2: p1_regrets_2, 3: p1_regrets_3}
    regrets2 = {1: p2_regrets_1, 2: p2_regrets_2, 3: p2_regrets_3}

    return game_1_current_value, regrets1, regrets2
