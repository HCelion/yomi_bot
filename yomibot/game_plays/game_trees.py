import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from yomibot.data.card_data import (
    penny_standard_payout,
    penny_non_standard_payout,
    penny_opponent_standard_payout,
    penny_non_standard_payout_opponent,
)
from yomibot.graph_models.helpers import PennyReservoir, invert_payout_dictionary
from yomibot.data.stochastic_game_regrets import get_optimal_policy
from yomibot.generic_code.penny.trees import PennyNode, PennyGame


def extract_regret_updates(states):
    regret_history = {1: [], 2: [], 3: []}
    for state in [1, 2, 3]:
        sub_states = [s for s in states if s.state_label == state]
        regret_history[state] = [
            {
                choice: float(regret)
                for choice, regret in zip(event["my_hand"].choices, event.regret)
            }
            for event in sub_states
        ]
    return regret_history


def get_new_policy(regrets):
    total_pos_regret = sum([positive_part(val) for val in regrets.values()])
    new_probs = {}
    for action, regret in regrets.items():
        if action in ["Even", "Odd"]:
            if total_pos_regret == 0:
                new_probs[action] = 1 / 2
            else:
                new_probs[action] = positive_part(regret) / total_pos_regret
    return new_probs


def extract_average_regrets(regret_history):
    if len(regret_history) == 0:
        return {"Even": 0, "Odd": 0}
    num = 0
    avg_regrets = {"Even": 0, "Odd": 0}
    for item in regret_history:
        avg_regrets["Even"] += item["Even"]
        avg_regrets["Odd"] += item["Odd"]
        num += 1
    avg_regrets["Even"] /= num
    avg_regrets["Odd"] /= num
    return avg_regrets


def positive_part(value):
    if value > 0:
        return value
    return 0.0


def extract_alpha_history(play_history, player):
    all_play_histories = []
    for state in [1, 2, 3]:
        state_history = (
            pd.DataFrame(play_history[state])
            .assign(avg_even=lambda x: x["Even"].cumsum() / x["Even"].count())
            .drop(columns=["Even", "Odd"])
            .assign(epoch=lambda x: np.arange(len(x)))
            .assign(state=state)
        )
        if player == 0:
            state_history = state_history.rename(columns={"avg_even": "alpha"})
        else:
            state_history = state_history.rename(columns={"avg_even": "beta"})
        all_play_histories.append(state_history)
    return pd.concat(all_play_histories)


def get_alpha_beta_history(play_history1, play_history2):
    p1_alphas = extract_alpha_history(play_history1, 0)
    p2_betas = extract_alpha_history(play_history2, 1)
    return p1_alphas.merge(p2_betas, on=["epoch", "state"])[
        ["epoch", "state", "alpha", "beta"]
    ]


def plot_penny(df, optimal_policy):
    from pylab import mpl, plt

    plt.style.use("bmh")
    mpl.rcParams["font.family"] = "serif"

    plt.style.use("bmh")
    mpl.rcParams["font.family"] = "serif"
    states = [1, 2, 3]
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    for i, state in enumerate(states):
        row = i

        state_df = df.query(f"state == {state}")

        axs[row].plot(
            state_df["alpha"],
            state_df["beta"],
            label="Alpha vs Beta",
            marker=".",
            alpha=0.3,
        )
        axs[row].scatter(
            optimal_policy[state]["alpha"],
            optimal_policy[state]["beta"],
            color="red",
            label="Nash Equilibrium",
            marker="x",
        )
        axs[row].set_title(f"State {state}")
        axs[row].set_xlabel("Alpha")
        axs[row].set_ylabel("Beta")
        axs[row].set_xlim(0, 1)
        axs[row].set_ylim(0, 1)
        axs[row].grid(True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # neutral_option = {'Even':0.5, 'Odd':0.5}
    # standard_model = model_dict = {state:neutral_option for state in [1,2,3]}

    uneven_option = {"Even": 0.2, "Odd": 0.8}
    standard_model = {state: uneven_option for state in [1, 2, 3]}
    model_dict = {0: standard_model, 1: standard_model}

    payout_dictionary = {
        1: (penny_standard_payout, penny_opponent_standard_payout),
        # 2: (penny_standard_payout, penny_opponent_standard_payout),
        3: (penny_standard_payout, penny_opponent_standard_payout),
        2: (penny_non_standard_payout, penny_non_standard_payout_opponent),
        # 3:(penny_non_standard_payout,penny_non_standard_payout_opponent)
    }

    optimal_policy = get_optimal_policy(payout_dictionary)

    reservoir_dict = {
        0: PennyReservoir("first_reservoir", payout_dictionary=payout_dictionary),
        1: PennyReservoir(
            "second_reservoir",
            payout_dictionary=invert_payout_dictionary(payout_dictionary),
        ),
    }

    regret_history1 = {1: [], 2: [], 3: []}
    regret_history2 = {1: [], 2: [], 3: []}
    play_history1 = {1: [], 2: [], 3: []}
    play_history2 = {1: [], 2: [], 3: []}

    for i in tqdm(range(100000)):
        for reservoir in reservoir_dict.values():
            reservoir.clear_reservoir()

        if i == 0:
            uneven_option = {"Even": 0.2, "Odd": 0.8}
            standard_model = {state: uneven_option for state in [1, 2, 3]}
            model_dict = {0: standard_model, 1: standard_model}
            for state in [1, 2, 3]:
                play_history1[state].append(deepcopy(model_dict[0][state]))
                play_history2[state].append(deepcopy(model_dict[1][state]))
        else:
            model1_dict = {}
            model2_dict = {}
            for state in [1, 2, 3]:
                model1_dict[state] = get_new_policy(
                    extract_average_regrets(regret_history1[state])
                )
                model2_dict[state] = get_new_policy(
                    extract_average_regrets(regret_history2[state])
                )
                play_history1[state].append(deepcopy(model1_dict[state]))
                play_history2[state].append(deepcopy(model2_dict[state]))
            model_dict = {0: model1_dict, 1: model2_dict}

        game = PennyGame(
            payout_dictionary=payout_dictionary,
            model_dict=model_dict,
            reservoir_dict=reservoir_dict,
        )
        node = PennyNode(
            state=1, game=game, player_0_action=None, player_1_action=None, player=0
        )
        node.get_regrets()

        states1 = reservoir_dict[0].sample(100)
        states2 = reservoir_dict[1].sample(100)
        regret1_update = extract_regret_updates(states1)
        regret2_update = extract_regret_updates(states2)
        for state in [1, 2, 3]:
            regret_history1[state] += regret1_update[state]
            regret_history2[state] += regret2_update[state]

    result_policy = get_alpha_beta_history(play_history1, play_history2)

    plot_penny(result_policy, optimal_policy)
