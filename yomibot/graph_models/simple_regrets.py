from copy import deepcopy
import pandas as pd
import numpy as np
from yomibot.graph_models.helpers import get_nash_equilibria
from yomibot.data.card_data import (
    get_penny_regrets,
    penny_standard_payout,
    penny_opponent_standard_payout,
    penny_non_standard_payout,
    penny_non_standard_payout_opponent,
)

choices = ["Even", "Odd"]


def positive_part(value):
    if value > 0:
        return value
    return 0.0


def get_new_policy(regrets):
    total_pos_regret = sum([positive_part(val) for val in regrets.values()])
    new_probs = {}
    for action, regret in regrets.items():
        if action in choices:
            if total_pos_regret == 0:
                new_probs[action] = 1 / 2
            else:
                new_probs[action] = positive_part(regret) / total_pos_regret
    return new_probs


def measure_regret_update(self_policy, other_policy, payout_function):
    choices = ["Even", "Odd"]
    regret_updates = get_penny_regrets(
        choices, self_policy, other_policy, payout_function
    )
    return {"Even": regret_updates[0][0].item(), "Odd": regret_updates[1][0].item()}


def get_average_play_style(play_history):
    total_plays = len(play_history)
    total_even = sum(ph["Even"] for ph in play_history)
    return {
        "Even": total_even / total_plays,
        "Odd": (total_plays - total_even) / total_plays,
    }


def get_latest_regret(regret_history, sample=False, sample_size=1000):
    if not sample:
        total_observations = len(regret_history)
        avg_even = sum(ph["Even"] for ph in regret_history) / total_observations
        avg_odd = sum(ph["Odd"] for ph in regret_history) / total_observations
        total_plus = positive_part(avg_even) + positive_part(avg_odd)
    else:
        if sample_size > len(regret_history):
            sample_size = len(regret_history)
        regret_sample = np.random.choice(regret_history, sample_size, replace=False)
        total_observations = len(regret_sample)
        avg_even = sum(ph["Even"] for ph in regret_sample) / total_observations
        avg_odd = sum(ph["Odd"] for ph in regret_sample) / total_observations
        total_plus = positive_part(avg_even) + positive_part(avg_odd)

    if total_plus == 0:
        return {"Even": 1 / 2, "Odd": 1 / 2}
    return {
        "Even": positive_part(avg_even) / total_plus,
        "Odd": positive_part(avg_odd) / total_plus,
    }


A = np.array([[4, -1], [-1, 1]])
player_1_optimum, player_2_optimum = get_nash_equilibria(A)


starting_regrets1 = {"Even": 2, "Odd": 1, "epoch": -1}
starting_regrets2 = {"Even": 1, "Odd": 2, "epoch": -1}
# player_1_payout = penny_standard_payout
# player_2_payout = penny_opponent_standard_payout
player_1_payout = penny_non_standard_payout
player_2_payout = penny_non_standard_payout_opponent

regret_update_history1 = [starting_regrets1]
regret_update_history2 = [starting_regrets2]
player_1_regrets = deepcopy(starting_regrets1)
player_2_regrets = deepcopy(starting_regrets2)
player_1_play_history = [{**get_new_policy(starting_regrets1), **{"epoch": -1}}]
player_2_play_history = [{**get_new_policy(starting_regrets2), **{"epoch": -1}}]

for i in range(10000):
    self_policy = get_latest_regret(regret_update_history1, sample=True, sample_size=1000)
    other_policy = get_latest_regret(
        regret_update_history2, sample=True, sample_size=1000
    )
    player_1_average = get_average_play_style(player_1_play_history)
    player_2_average = get_average_play_style(player_2_play_history)
    self_policy["epoch"] = i
    other_policy["epoch"] = i
    player_1_play_history.append(self_policy)
    player_2_play_history.append(other_policy)
    regret_updates1 = measure_regret_update(self_policy, other_policy, player_1_payout)
    regret_updates2 = measure_regret_update(other_policy, self_policy, player_2_payout)
    regret_updates1["epoch"] = i
    regret_updates2["epoch"] = i
    regret_update_history1.append(regret_updates1)
    regret_update_history2.append(regret_updates2)

play_history = (
    pd.DataFrame(player_1_play_history)
    .assign(even=lambda x: x["Even"].cumsum())
    .assign(odd=lambda x: x["Odd"].cumsum())
    .assign(alpha=lambda x: x.even / (x.even + x.odd))
    .merge(
        (
            pd.DataFrame(player_2_play_history)
            .assign(even=lambda x: x["Even"].cumsum())
            .assign(odd=lambda x: x["Odd"].cumsum())
            .assign(beta=lambda x: x.even / (x.even + x.odd))[["epoch", "beta"]]
        ),
        on="epoch",
    )
)


play_history.plot(x="alpha", y="beta")


(
    pd.DataFrame(regret_update_history1)
    .assign(even=lambda x: x["Even"].cumsum())
    .assign(odd=lambda x: x["Odd"].cumsum())
    .assign(
        alpha=lambda x: np.where(x.even < 0, 0, x.even / (x.even + np.maximum(0, x.odd)))
    )
    # .plot(x='epoch', y = 'alpha')
)
