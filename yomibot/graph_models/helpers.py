import os
from collections import Counter
from torch import optim
import numpy as np
import pandas as pd
import ternary
import pytorch_lightning as pl
from pylab import mpl, plt
from pytorch_lightning.callbacks import ModelCheckpoint
from yomibot.common.paths import data_path
from yomibot.data.card_data import PennyData
from yomibot.data.helpers import flatten_dict, unflatten_dict

checkpoint_callback = ModelCheckpoint(save_top_k=0)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):  # pylint: disable=(protected-access)
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect metrics at the end of each training epoch
        self.metrics.append(trainer.callback_metrics.copy())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect metrics at the end of each validation epoch
        # print(trainer.callback_metrics.copy())
        # self.metrics.append(trainer.callback_metrics.copy())
        pass


def get_nash_equilibria(A, return_value=False):
    import nashpy as nash

    rps = nash.Game(A, -A)
    eqs = rps.support_enumeration()
    player_1, player_2 = list(eqs)[0]
    order = ("Rock", "Paper", "Scissors")
    optimum_1 = {action: prob for action, prob in zip(order, player_1)}
    optimum_2 = {action: prob for action, prob in zip(order, player_2)}

    if return_value:
        p1_value = np.array(player_1).T @ A @ np.array(player_2)
        return optimum_1, optimum_2, p1_value
    return optimum_1, optimum_2


def plot_model_history(model_history, equilibrium_point):
    history_df = pd.DataFrame(model_history)
    fig = plt.figure(figsize=(10, 6))

    # Define colors for Rock, Paper, and Scissors
    colors = {"Rock": "red", "Paper": "blue", "Scissors": "green"}

    # Plot the history lines with the defined colors
    plt.plot(history_df.index, history_df["Rock"], label="Rock", color=colors["Rock"])
    plt.plot(history_df.index, history_df["Paper"], label="Paper", color=colors["Paper"])
    plt.plot(
        history_df.index,
        history_df["Scissors"],
        label="Scissors",
        color=colors["Scissors"],
    )
    plt.ylim(0, 1)

    # Plot the equilibrium points with the same colors
    plt.axhline(
        y=equilibrium_point["Rock"],
        color=colors["Rock"],
        linestyle=":",
        label="Equilibrium Rock",
    )
    plt.axhline(
        y=equilibrium_point["Paper"],
        color=colors["Paper"],
        linestyle=":",
        label="Equilibrium Paper",
    )
    plt.axhline(
        y=equilibrium_point["Scissors"],
        color=colors["Scissors"],
        linestyle=":",
        label="Equilibrium Scissors",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.title("Train History")
    plt.legend()

    # Return the figure
    return fig


def plot_model_history_with_mse(model_history, equilibrium_point):
    mse_rock = []
    mse_paper = []
    mse_scissors = []

    for record in model_history:
        mse_rock.append((record["Rock"] - equilibrium_point["Rock"]) ** 2)
        mse_paper.append((record["Paper"] - equilibrium_point["Paper"]) ** 2)
        mse_scissors.append((record["Scissors"] - equilibrium_point["Scissors"]) ** 2)

    overall_mse = [r + p + s for r, p, s in zip(mse_rock, mse_paper, mse_scissors)]
    epochs = [record["epoch"] for record in model_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the MSE values
    ax.plot(epochs, mse_rock, label="MSE Rock", color="red", linestyle="--")
    ax.plot(epochs, mse_paper, label="MSE Paper", color="blue", linestyle="--")
    ax.plot(epochs, mse_scissors, label="MSE Scissors", color="green", linestyle="--")
    ax.plot(epochs, overall_mse, label="Overall MSE", color="purple", linestyle="-")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Mean Squared Error Over Time")
    ax.legend(loc="upper right")

    plt.show()


def plot_model_history_ternary(model_history, equilibrium_point):
    # Convert the equilibrium point dictionary to a list of lists
    equilibrium_points = [
        [
            equilibrium_point["Rock"],
            equilibrium_point["Paper"],
            equilibrium_point["Scissors"],
        ]
    ]

    history_df = pd.DataFrame(model_history)

    # Extract probabilities for Rock, Paper, and Scissors
    points = history_df[["Rock", "Paper", "Scissors"]].values.tolist()

    # Initialize the ternary plot
    scale = 1.0
    fig, ax = plt.subplots(figsize=(10, 8))
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    tax.boundary(linewidth=1.0)  # Thinner boundary lines
    tax.gridlines(color="blue", multiple=0.1, linewidth=0.5)  # Thinner grid lines

    # Plot the points with lower opacity and thinner lines
    tax.scatter(
        points, marker="o", color="red", label="Strategy", s=10, alpha=0.5
    )  # s=10 for smaller markers, alpha=0.5 for lower opacity
    tax.plot(
        points, color="red", linewidth=0.5, alpha=0.5
    )  # Thinner lines and lower opacity

    # Mark the starting point with a different marker and color
    if points:
        tax.scatter(
            [points[0]], marker="s", color="blue", label="Start", s=50
        )  # s=50 for larger marker

    # Plot the equilibrium points last to ensure they are on top
    tax.scatter(
        equilibrium_points, marker="x", color="green", label="Equilibrium", s=50
    )  # s=50 for larger markers

    # Set axis labels at the corners
    tax.left_corner_label("Rock", offset=0.16)
    tax.right_corner_label("Paper", offset=0.16)
    tax.top_corner_label("Scissors", offset=0.16, verticalalignment="bottom")
    tax.set_title("Strategy Space History")

    # Add legend
    tax.legend()

    # Return the figure
    return fig


def parse_log_item(log, model_number):
    return {
        "Rock": float(log[f"model_{model_number}_Rock"]),
        "Scissors": float(log[f"model_{model_number}_Scissors"]),
        "Paper": float(log[f"model_{model_number}_Paper"]),
        "epoch": int(log["epoch"]),
    }


def parse_freq_log_item(log, model_number):
    return {
        "Rock": float(log[f"freq_model_{model_number}_Rock"]),
        "Scissors": float(log[f"freq_model_{model_number}_Scissors"]),
        "Paper": float(log[f"freq_model_{model_number}_Paper"]),
        "epoch": int(log["epoch"]),
    }


def parse_q_log_item(log, model_number):
    return {
        "Rock": float(log[f"q_{model_number}_Rock"]),
        "Scissors": float(log[f"q_{model_number}_Scissors"]),
        "Paper": float(log[f"q_{model_number}_Paper"]),
        "epoch": int(log["epoch"]),
    }


class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.is_full = False

    def append(self, item):
        self.buffer[self.head] = item
        if self.is_full:
            self.tail = (self.tail + 1) % self.size
        self.head = (self.head + 1) % self.size
        self.is_full = self.head == self.tail

    def add_set(self, items):
        for item in items:
            self.append(item)

    def read(self):
        if self.head == self.tail and not self.is_full:
            raise IndexError("Buffer is empty")
        item = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.size
        self.is_full = False
        return item

    def __repr__(self):
        if self.is_full:
            return f"CircularBuffer({self.buffer})"
        if self.head >= self.tail:
            return f"CircularBuffer({self.buffer[self.tail:self.head]})"
        return f"CircularBuffer({self.buffer[self.tail:] + self.buffer[:self.head]})"


def empirical_frequencies(list_items):
    return {
        action: val / len(list_items) for action, val in dict(Counter(list_items)).items()
    }


def plot_penny(df):
    states = df["state"].unique()
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, state in enumerate(states):
        row = i // 2
        col = i % 2
        state_data = df[df["state"] == state]

        axs[row, col].plot(
            state_data["alpha"], state_data["beta"], label="Alpha vs Beta", marker="o"
        )
        axs[row, col].plot(
            state_data["alpha_freq"],
            state_data["beta_freq"],
            label="Alpha_freq vs Beta_freq",
            marker="x",
        )

        axs[row, col].set_title(f"State {state}")
        axs[row, col].set_xlabel("Alpha / Alpha_freq")
        axs[row, col].set_ylabel("Beta / Beta_freq")
        axs[row, col].set_xlim(0, 1)
        axs[row, col].set_ylim(0, 1)
        axs[row, col].legend()
        axs[row, col].grid(True)

    # Hide the empty subplot if the number of states is less than 4
    if len(states) < 4:
        fig.delaxes(axs[1, 1])

    plt.tight_layout()
    return fig


def turn_to_penny_df(logged_metrics):
    state_records = []
    for metric in logged_metrics:
        epoch = int(metric["epoch"])
        unflattened_metric = unflatten_dict(metric)
        for player in [1, 2]:
            policy = "alpha" if player == 1 else "beta"
            alphas = [
                {
                    "state": s,
                    "player": player,
                    "policy": policy,
                    "policy_even": float(
                        unflattened_metric[f"model_{player}"][str(s)]["Even"]
                    ),
                    "actual_even_play": float(
                        unflattened_metric[f"play_hist{player}"][str(s)][policy]
                    ),
                    "regret_update_even": float(
                        unflattened_metric[f"emp_regret_{player}"][str(s)]["Even"]
                    ),
                    "regret_update_odd": float(
                        unflattened_metric[f"emp_regret_{player}"][str(s)]["Odd"]
                    ),
                    "regret_even_trained": float(
                        unflattened_metric[f"q_values_{player}"][str(s)]["Even"]
                    ),
                    "regret_odd_trained": float(
                        unflattened_metric[f"q_values_{player}"][str(s)]["Odd"]
                    ),
                    "epoch": epoch,
                }
                for s in [1, 2, 3]
            ]
            state_records.extend(alphas)
    df = pd.DataFrame.from_records(state_records)
    return (
        df.assign(
            mean_policy=lambda x: x.groupby(["state", "player"]).policy_even.transform(
                "cumsum"
            )
            / (x.groupby(["state", "player"]).policy_even.transform("cumcount") + 1),
            even_mean_regret=lambda x: x.groupby(
                ["state", "player"]
            ).regret_update_even.transform("cumsum")
            / (
                x.groupby(["state", "player"]).regret_update_even.transform("cumcount")
                + 1
            ),
            odd_mean_regret=lambda x: x.groupby(
                ["state", "player"]
            ).regret_update_odd.transform("cumsum")
            / (
                x.groupby(["state", "player"]).regret_update_odd.transform("cumcount") + 1
            ),
        )
        .assign(even_train_diff=lambda x: x.even_mean_regret - x.regret_even_trained)
        .assign(odd_train_diff=lambda x: x.odd_mean_regret - x.regret_odd_trained)
        .assign(
            policy_error=lambda x: np.logical_or(
                np.sign(x.even_mean_regret) != np.sign(x.regret_even_trained),
                np.sign(x.odd_mean_regret) != np.sign(x.regret_odd_trained),
            )
        )
    )


class PennyReservoir:
    def __init__(self, reservoir_name, payout_dictionary=None):
        self.reservoir_name = reservoir_name
        self.file_folder = data_path / reservoir_name
        self.file_folder.mkdir(parents=True, exist_ok=True)
        self.payout_dictionary = payout_dictionary
        self.num_files = 0
        self.summary = None
        self.generate_summary()

    def __repr__(self):
        return "PennyReservoir(" + self.reservoir_name + "," + str(len(self)) + ")"

    def __len__(self):
        return len(self.summary)

    def generate_summary(self):
        files = os.listdir(self.file_folder)
        summary_files = []
        num_files = 0
        for file in files:
            if file.endswith(".pq"):
                df = pd.read_parquet(self.file_folder / file)
                df["file_name"] = file
                summary_files.append(df[["file_name", "file_index", "weight"]])
                num_files += 1
        if len(summary_files) > 0:
            summary = pd.concat(summary_files)
        else:
            summary = pd.DataFrame(columns=["file_name", "file_index", "weight"])
        self.summary = summary
        self.num_files = num_files

    def store_data(self, data_items):
        serialisation = pd.DataFrame([s.serialise() for s in data_items])
        serialisation["file_index"] = np.arange(len(serialisation))
        file_number = self.num_files + 1
        file_name = "file_batch_" + str(file_number) + ".pq"
        serialisation.to_parquet(self.file_folder / file_name, index=False)
        serialisation["file_name"] = file_name
        self.summary = pd.concat(
            [self.summary, serialisation[["file_name", "file_index", "weight"]]]
        )
        self.num_files += 1

    def sample(self, num_items):
        if num_items > len(self.summary):
            file_names = self.summary.file_name.unique()
            all_dfs = pd.concat(
                pd.read_parquet(self.file_folder / fn) for fn in file_names
            )
        else:
            sampled_files = self.summary.sample(
                num_items, weights=self.summary.weight, replace=False
            )
            all_samples = []
            for file_name, data in sampled_files.groupby("file_name"):
                files_to_keep = data.file_index
                sample = pd.read_parquet(self.file_folder / file_name).query(
                    "file_index.isin(@files_to_keep)"
                )
                all_samples.append(sample)
            all_dfs = pd.concat(all_samples)

        # Shuffle and to dict
        serialised_data = all_dfs.sample(frac=1).to_dict("records")
        data_items = [
            PennyData.deserialise(data, payout_dictionary=self.payout_dictionary)
            for data in serialised_data
        ]
        return data_items

    def clear_reservoir(self):
        files = os.listdir(self.file_folder)
        for file in files:
            if file.endswith(".pq"):
                os.remove(self.file_folder / file)
        self.generate_summary()


def invert_payout_dictionary(payout_dictionary):
    return {
        state: (opp_payout, payout)
        for state, (payout, opp_payout) in payout_dictionary.items()
    }


def get_empirical_ratios(states, descriptor):
    emps = empirical_frequencies([(s.state_label, s.self_action) for s in states])
    return {
        s: {
            descriptor: emps.get((s, "Even"), 0)
            / (emps.get((s, "Even"), 0) + emps.get((s, "Odd"), 0))
        }
        for s in [1, 2, 3]
    }


def get_empirical_regrets(states):
    state_summaries = pd.DataFrame(
        [
            {"state": s.state_label, "regret": s.regret, "choices": s["my_hand"].choices}
            for s in states
        ]
    )
    aggs = (
        state_summaries.assign(choice_0=lambda x: x.choices.apply(lambda y: y[0]))
        .assign(regret_0=lambda x: x.regret.apply(lambda y: float(y[0][0])))
        .assign(regret_1=lambda x: x.regret.apply(lambda y: float(y[1][0])))
        .drop(columns=["regret", "choices"])
        .assign(
            regret_even=lambda x: np.where(x.choice_0 == "Even", x.regret_0, x.regret_1)
        )
        .assign(
            regret_odd=lambda x: np.where(x.choice_0 == "Odd", x.regret_0, x.regret_1)
        )
        .drop(columns=["choice_0", "regret_0", "regret_1"])
        .rename(columns={"regret_even": "Even", "regret_odd": "Odd"})
        .melt(id_vars="state", var_name="choice", value_name="regret")
        .assign(state_choice=lambda x: x.state.astype(str) + "." + x.choice)
        .groupby(["state_choice"], as_index=False)
        .agg(regret=("regret", "mean"))
        .to_dict("records")
    )
    return {rec["state_choice"]: round(rec["regret"], 3) for rec in aggs}
