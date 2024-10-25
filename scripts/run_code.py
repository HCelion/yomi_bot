import math
import pandas as pd
import numpy as np
import random
from torch_geometric.data import Batch
from torch.nn.functional import mse_loss, softmax, sigmoid
import pytorch_lightning as pl
import torch
from torch import optim
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from yomibot.graph_models.base_encoder import RPSHandChoiceModel, RPSUtilityModel
from yomibot.graph_models.helpers import CosineWarmupScheduler
from yomibot.data.card_data import generate_rps_sample, CardDataset
from yomibot.common import paths
from yomibot.graph_models.models import (
    RPSSuccessModel,
    RPSChoiceModel,
    RPSAvgActionModel,
    RPSCurrentActionModel,
)


utility_model = RPSSuccessModel(hidden_dim=3, final_dim=2, num_layers=1, dropout=0)
batch = Batch.from_data_list([generate_rps_sample() for _ in range(10)])

dataset_size = 1000
dataset = CardDataset([generate_rps_sample() for _ in range(dataset_size)])

train_set = dataset[: math.floor(dataset_size * 0.8)]
test_set = dataset[math.floor(dataset_size * 0.8) :]

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True)

model_name = "rps_model"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

trainer = pl.Trainer(
    default_root_dir=paths.model_artifact_path / model_name,
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=10,
    enable_progress_bar=False,
    gradient_clip_val=1,
    fast_dev_run=False,
)

trainer.fit(utility_model, train_loader)

test_batch = Batch.from_data_list([data for data in test_set])
utility_model.predict_step(test_batch, test_batch.batch_dict)

model = RPSChoiceModel(
    hidden_dim=8,
    num_layers=3,
    final_dim=5,
    dropout=0.1,
    input_bias=True,
    bias=True,
    lr=0.1,
    weight_decay=0.01,
)

dataset_size = 1000
dataset = CardDataset([generate_rps_sample() for _ in range(dataset_size)])

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

model_name = "rps_model"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

trainer = pl.Trainer(
    default_root_dir=paths.model_artifact_path / model_name,
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=10,
    enable_progress_bar=False,
    gradient_clip_val=1,
    fast_dev_run=False,
)

trainer.fit(model, train_loader)

test_batch = Batch.from_data_list([data for data in test_set])
model.predict_step(test_batch)

avg_policy_model = RPSAvgActionModel(hidden_dim=5, final_dim=3, num_layers=2)
rps_current_action_model = RPSCurrentActionModel(
    hidden_dim=5,
    final_dim=3,
    num_layers=2,
    avg_policy_model=avg_policy_model,
    delta_win=1,
    delta_lose=1,
    lr=0.01,
)

dataset_size = 10000

my_model = [("Rock", 0.5), ("Paper", 0.2), ("Scissors", 0.3)]
opponent_model = [("Rock", 0), ("Paper", 0), ("Scissors", 1)]

dataset = CardDataset(
    [
        generate_rps_sample(self_model=my_model, opponent_model=opponent_model)
        for _ in range(dataset_size)
    ]
)

train_set = dataset[: math.floor(dataset_size * 0.8)]
test_set = dataset[math.floor(dataset_size * 0.8) :]
from collections import Counter

Counter([data.opponent_action for data in train_set])
counter = Counter([data.self_action for data in train_set])
expectations_avg_policy = {
    key: val / sum(val for val in counter.values()) for key, val in counter.items()
}

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

model_name = "rps_model"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

avg_policy_trainer = pl.Trainer(
    default_root_dir=paths.model_artifact_path / "avg_policy",
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=1,
    enable_progress_bar=False,
    gradient_clip_val=1,
    fast_dev_run=False,
)

current_policy_trainer = pl.Trainer(
    default_root_dir=paths.model_artifact_path / "current_policy",
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=1,
    enable_progress_bar=False,
    gradient_clip_val=1,
    fast_dev_run=False,
)

avg_policy_trainer.fit(avg_policy_model, train_loader)
current_policy_trainer.fit(rps_current_action_model, train_loader)

data = Batch.from_data_list([data for data in test_set])
rps_current_action_model.avg_policy_model.predict_step(data)

rps_current_action_model.avg_policy_model.generate_prob_model()
rps_current_action_model.predict_step(data)
