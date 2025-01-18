import math
import numpy as np
import logging
import torch
import ternary
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from yomibot.graph_models.models import (
    RPSSuccessModel,
    RPSRandomWWeightModel,
    RPSWolfModel,
    RPSChoiceModel,
    RPSAvgActionModel,
    RPSCurrentActionModel,
)
from yomibot.graph_models.helpers import get_nash_equilibria
from yomibot.data.card_data import (
    generate_rps_sample,
    CardDataset,
    rps_non_standard_payout,
    rps_standard_payout,
    rps_non_standard_payout_opponent,
)
from yomibot.graph_models.helpers import (
    CosineWarmupScheduler,
    MetricsCallback,
    plot_model_history,
    plot_model_history_with_mse,
    plot_model_history_ternary,
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

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, train_loader


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


def generate_deep_wolf(lr=0.1, avg_reduction=0.5, delta_win=0.001, delta_lose=0.002):
    avg_policy_model = RPSAvgActionModel(
        hidden_dim=5, final_dim=3, num_layers=1, lr=avg_reduction * lr
    )

    rps_current_action_model = RPSCurrentActionModel(
        hidden_dim=5,
        final_dim=3,
        num_layers=1,
        lr=lr,
        avg_policy_model=avg_policy_model,
        delta_win=delta_win,
        delta_lose=delta_lose,
    )
    return rps_current_action_model


def generate_wolf_trainers(model_name, max_epochs=1):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    avg_policy_trainer = pl.Trainer(
        default_root_dir=paths.model_artifact_path / (model_name + "_avg_policy"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        enable_progress_bar=False,
        gradient_clip_val=1,
        fast_dev_run=False,
    )

    current_policy_trainer = pl.Trainer(
        default_root_dir=paths.model_artifact_path / (model_name + "_current_policy"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        enable_progress_bar=False,
        gradient_clip_val=1,
        fast_dev_run=False,
    )
    return {
        "avg_policy": avg_policy_trainer,
        "current_policy": current_policy_trainer,
    }


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


def train_wolf_model_deep(
    payout_function1,
    payout_function2,
    num_iterations=10,
    sample_size=1000,
    lr=0.1,
    avg_reduction=0.5,
    value_reduction=0.5,
    max_epochs=1,
    delta_win=0.001,
    delta_lose=0.002,
):
    model1 = generate_deep_wolf(lr=lr, delta_win=delta_win, delta_lose=delta_lose)
    model2 = generate_deep_wolf(lr=lr, delta_win=delta_win, delta_lose=delta_lose)

    model1_history = []
    model2_history = []
    model1_avg_history = []
    model2_avg_history = []

    m1_report, m2_report = gen_report(-1, model1, model2)
    model1_history.append(m1_report)
    model2_history.append(m2_report)
    model1_avg_history.append(
        {**model1.avg_policy_model.generate_prob_model(as_dict=True), "epoch": -1}
    )
    model2_avg_history.append(
        {**model2.avg_policy_model.generate_prob_model(as_dict=True), "epoch": -1}
    )

    for epoch in tqdm(range(num_iterations)):
        model1_set, m1_train = generate_rps_dataset_with_loaders(
            payout_function=payout_function1,
            self_model=model1,
            opponent_model=model2,
            N=sample_size,
        )

        model2_set, m2_train = generate_rps_dataset_with_loaders(
            payout_function=payout_function2,
            self_model=model2,
            opponent_model=model1,
            N=sample_size,
        )

        m1_trainers = generate_wolf_trainers("wolf_m1", max_epochs=max_epochs)
        m2_trainers = generate_wolf_trainers("wolf_m2", max_epochs=max_epochs)

        m1_trainers["avg_policy"].fit(model1.avg_policy_model, m1_train)
        m2_trainers["avg_policy"].fit(model2.avg_policy_model, m2_train)

        m1_trainers["current_policy"].fit(model1, m1_train)
        m2_trainers["current_policy"].fit(model2, m2_train)

        m1_report, m2_report = gen_report(epoch, model1, model2)
        model1_avg_history.append(
            {**model1.generate_prob_model(as_dict=True), "epoch": epoch}
        )
        model2_avg_history.append(
            {**model2.generate_prob_model(as_dict=True), "epoch": epoch}
        )
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
        num_iterations=50000,
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

    ## WoLF implementation
    A = np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]])
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
    ) = train_wolf_model_deep(
        payout_function1=payout_function1,
        payout_function2=payout_function2,
        num_iterations=200,
        sample_size=30,
        lr=0.05,
        avg_reduction=1,
        max_epochs=1,
        delta_win=0.1,
        delta_lose=1,
    )

    model1.avg_policy_model.generate_prob_model()
    model2.avg_policy_model.generate_prob_model()

    fig = plot_model_history(model1_history, player_1_optimum)
    fig = plot_model_history(model2_history, player_2_optimum)

    fig = plot_model_history_ternary(model1_history, player_1_optimum)
    fig = plot_model_history_ternary(model2_history, player_2_optimum)

    fig = plot_model_history_with_mse(model1_avg_history, player_1_optimum)
    fig = plot_model_history_with_mse(model2_avg_history, player_2_optimum)
