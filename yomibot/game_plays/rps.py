import math
import numpy as np
import logging
import torch
import ternary
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from yomibot.graph_models.models import (
    RPSChoiceModel,
    RPSSuccessModel,
    RPSRandomWWeightModel,
    RPSWolfModel,
)
from yomibot.graph_models.helpers import get_nash_equilibria
from yomibot.data.card_data import (
    generate_rps_sample,
    CardDataset,
    rps_non_standard_payout,
    rps_standard_payout,
    rps_non_standard_payout_opponent,
)
from yomibot.common import paths
import pandas as pd
from tqdm import tqdm
from pylab import mpl, plt
import warnings


plt.style.use("bmh")
mpl.rcParams["font.family"] = "serif"

warnings.filterwarnings("ignore")

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)


def generate_rps_dataset(
    payout_function,
    self_model=None,
    opponent_model=None,
    N=1000,
):
    if self_model:
        state_model = self_model.generate_prob_model()
    else:
        state_model = [("Rock", 0.333), ("Paper", 0.333), ("Scissors", 0.333)]

    if opponent_model:
        opp_state_model = opponent_model.generate_prob_model()
    else:
        opp_state_model = [("Rock", 0.333), ("Paper", 0.333), ("Scissors", 0.333)]

    dataset = CardDataset(
        [
            generate_rps_sample(
                payout_function=payout_function,
                self_model=state_model,
                opponent_model=opp_state_model,
            )
            for _ in range(N)
        ]
    )

    return dataset


def generate_rps_dataset_with_loaders(
    payout_function,
    self_model=None,
    opponent_model=None,
    N=1000,
    batch_size=128,
    shuffle=True,
):
    dataset = generate_rps_dataset(
        payout_function=payout_function,
        self_model=self_model,
        opponent_model=opponent_model,
        N=N,
    )
    train_set = dataset[: math.floor(N * 0.8)]
    val_set = dataset[math.floor(N * 0.8) :]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return dataset, train_loader, val_loader


def gen_trainer(model_name):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return pl.Trainer(
        default_root_dir=paths.model_artifact_path / model_name,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=5,
        enable_progress_bar=False,
        gradient_clip_val=1,
        fast_dev_run=False,
        logger=False,
    )


def gen_model():
    utility_model = RPSSuccessModel(
        hidden_dim=3, final_dim=2, num_layers=1, num_heads=1, dropout=0
    )
    return RPSChoiceModel(
        utility_model=utility_model,
        hidden_dim=8,
        num_layers=3,
        final_dim=5,
        num_heads=2,
        dropout=0.0,
        input_bias=True,
        bias=True,
        lr=0.001,
        weight_decay=0.001,
    )


def gen_regret_model(eta=0.01):
    return RPSRandomWWeightModel(eta=eta)


def gen_wolf_model(alpha=0.01, delta_win=0.01, delta_lose=0.1, explore_rate=0.01):
    return RPSWolfModel(
        alpha=alpha, delta_win=delta_win, delta_lose=delta_lose, explore_rate=explore_rate
    )


def gen_report(epoch, model1, model2):
    m1_report = {key: val for key, val in model1.generate_prob_model(explore=False)}
    m1_report["epoch"] = epoch
    m2_report = {key: val for key, val in model2.generate_prob_model(explore=False)}
    m2_report["epoch"] = epoch
    return m1_report, m2_report


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


def train_rps_model(
    payout_function1, payout_function2, num_iterations=10, sample_size=1000
):
    model1 = gen_model()
    model2 = gen_model()

    model1_history = []
    model2_history = []

    m1_report, m2_report = gen_report(0, model1, model2)
    model1_history.append(m1_report)
    model2_history.append(m2_report)

    for epoch in tqdm(range(num_iterations)):
        model1_set, m1_train, m1_val = generate_rps_dataset_with_loaders(
            payout_function=payout_function1,
            self_model=model1,
            opponent_model=model2,
            N=sample_size,
        )
        model2_set, m2_train, m2_val = generate_rps_dataset_with_loaders(
            payout_function=payout_function2,
            self_model=model2,
            opponent_model=model1,
            N=sample_size,
        )

        trainer1 = gen_trainer("rps_model1")
        trainer2 = gen_trainer("rps_model2")
        trainer1_success = gen_trainer("rps_model1_success")
        trainer2_success = gen_trainer("rps_model2_success")

        trainer1.fit(model1, m1_train, m1_val)
        trainer2.fit(model2, m2_train, m2_val)
        trainer1_success.fit(model1.utility_model, m1_train, m1_val)
        trainer2_success.fit(model2.utility_model, m1_train, m1_val)

        m1_report, m2_report = gen_report(epoch, model1, model2)
        model1_history.append(m1_report)
        model2_history.append(m2_report)

    return model1, model2, model1_history, model2_history


def train_rps_regret_model(
    payout_function1,
    payout_function2,
    num_iterations=10,
    sample_size=1000,
    alpha=0.01,
    delta_win=0.01,
    delta_lose=0.1,
):
    model1 = gen_wolf_model(alpha=alpha, delta_win=delta_win, delta_lose=delta_lose)
    model2 = gen_wolf_model(alpha=alpha, delta_win=delta_win, delta_lose=delta_lose)

    model1_history = []
    model2_history = []
    model1_avg_history = []
    model2_avg_history = []

    m1_report, m2_report = gen_report(0, model1, model2)
    model1_history.append(m1_report)
    model1_avg_history.append({**model1.start_model, "epoch": 0})
    model2_avg_history.append({**model2.start_model, "epoch": 0})
    model2_history.append(m2_report)

    for epoch in tqdm(range(num_iterations)):
        model1_set = generate_rps_dataset(
            payout_function=payout_function1,
            self_model=model1,
            opponent_model=model2,
            N=sample_size,
        )

        model2_set = generate_rps_dataset(
            payout_function=payout_function2,
            self_model=model2,
            opponent_model=model1,
            N=sample_size,
        )

        model1.train(model1_set)
        model2.train(model2_set)

        m1_report, m2_report = gen_report(epoch, model1, model2)
        model1_avg_history.append({**model1.average_policy, "epoch": epoch})
        model2_avg_history.append({**model2.average_policy, "epoch": epoch})
        model1_history.append(m1_report)
        model2_history.append(m2_report)

    return (
        model1,
        model2,
        model1_history,
        model2_history,
        model1_avg_history,
        model2_avg_history,
    )


if __name__ == "__main__":
    # A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    A = np.array([[0, -1 / 2, 1], [1 / 2, 0, -1 / 2], [-1 / 2, 1 / 2, 0]])
    payout_function1 = rps_non_standard_payout
    payout_function2 = rps_non_standard_payout_opponent
    player_1_optimum, player_2_optimum = get_nash_equilibria(A)

    (
        model1,
        model2,
        model1_history,
        model2_history,
        model1_avg_history,
        model2_avg_history,
    ) = train_rps_regret_model(
        payout_function1=payout_function1,
        payout_function2=payout_function2,
        num_iterations=20000,
        sample_size=1,
        alpha=0.1,
        delta_win=0.001,
        delta_lose=0.002,
    )

    model1.generate_prob_model()
    model2.generate_prob_model()
    model1.average_policy
    model2.average_policy

    fig = plot_model_history_ternary(model1_history, player_1_optimum)
    fig = plot_model_history_ternary(model2_history, player_2_optimum)

    fig = plot_model_history_with_mse(model1_avg_history, player_1_optimum)
    fig = plot_model_history_with_mse(model2_avg_history, player_2_optimum)

    fig = plot_model_history(model1_history, player_1_optimum)
    fig = plot_model_history(model2_history, player_2_optimum)
