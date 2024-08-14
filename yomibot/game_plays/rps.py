import math
import logging
from random import choices
import torch
import numpy as np
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from collections import Counter
from yomibot.graph_models.models import RPSModel
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

plt.style.use("bmh")
mpl.rcParams["font.family"] = "serif"

import warnings

warnings.filterwarnings("ignore")

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)


def generate_rps_dataset(
    payout_function, model=None, N=1000, batch_size=128, shuffle=True
):
    if model:
        state_model = model.generate_prob_model()
    else:
        state_model = [("Rock", 0.333), ("Paper", 0.333), ("Scissors", 0.333)]
    dataset = CardDataset(
        [
            generate_rps_sample(
                payout_function=payout_function, opponent_model=state_model
            )
            for _ in range(N)
        ]
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
    return RPSModel(
        hidden_dim=8,
        num_layers=3,
        final_dim=5,
        num_heads=2,
        dropout=0.0,
        input_bias=True,
        bias=True,
        lr=0.0001,
        use_projection=True,
        weight_decay=0.001,
    )


def gen_report(epoch, model1, model2):
    m1_report = {key: val for key, val in model1.generate_prob_model()}
    m1_report["epoch"] = epoch
    m2_report = {key: val for key, val in model2.generate_prob_model()}
    m2_report["epoch"] = epoch
    return m1_report, m2_report


def plot_model_history(model_history, horizontal_lines):
    history_df = pd.DataFrame(model_history)
    fig = plt.figure(figsize=(10, 6))

    plt.plot(history_df.index, history_df["Rock"], label="Rock")
    plt.plot(history_df.index, history_df["Paper"], label="Paper")
    plt.plot(history_df.index, history_df["Scissors"], label="Scissors")
    plt.ylim(0, 1)

    for line in horizontal_lines:
        plt.axhline(y=line, color="r", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.title("Train History")
    plt.legend()

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
        model1_set, m1_train, m1_val = generate_rps_dataset(
            payout_function=payout_function1, model=model2, N=sample_size
        )
        model2_set, m2_train, m2_val = generate_rps_dataset(
            payout_function=payout_function2, model=model1, N=sample_size
        )

        trainer1 = gen_trainer("rps_model1")
        trainer2 = gen_trainer("rps_model2")

        trainer1.fit(model1, m1_train, m1_val)
        trainer2.fit(model2, m2_train, m2_val)

        m1_report, m2_report = gen_report(epoch, model1, model2)
        model1_history.append(m1_report)
        model2_history.append(m2_report)

    return model1, model2, model1_history, model2_history


A = np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]])
A

player_1_optimum, player_2_optimum = get_nash_equilibria(A)
player_1_optimum

model1, model2, model1_history, model2_history = train_rps_model(
    payout_function1=rps_non_standard_payout,
    payout_function2=rps_non_standard_payout_opponent,
    num_iterations=500,
    sample_size=100,
)

model1.generate_prob_model()
model2.generate_prob_model()

fig = plot_model_history(model1_history, player_1_optimum)
fig = plot_model_history(model2_history, player_2_optimum)
