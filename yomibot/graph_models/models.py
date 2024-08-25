import math
import pandas as pd
import numpy as np
import random
from torch_geometric.data import Batch
from torch.nn.functional import mse_loss
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


def clip_value(value):
    return max(0, min(value, 1))


class RPSSuccessModel(pl.LightningModule):
    def __init__(
        self,
        lr=0.1,
        weight_decay=0.01,
        warmup=10,
        max_iters=2000,
        **model_kwargs,
    ):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.undirected_transformer = ToUndirected()
        self.example_data = generate_rps_sample()
        self.model = RPSUtilityModel(**model_kwargs)

    def forward(  # pylint: disable=(too-many-locals
        self,
        data,
        mode="train",  # pylint: disable=(unused-argument
    ):
        data = self.undirected_transformer(data)

        x_dict, edge_index_dict, batch_dict, actual_utility = (
            data.x_dict,
            data.edge_index_dict,
            data.batch_dict,
            data.my_utility,
        )

        predicted_utility = self.model(
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )

        loss = mse_loss(predicted_utility.float(), target=actual_utility.float())
        max_pred = predicted_utility.max()

        return loss, max_pred

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):  # pylint: disable=(unused-argument)
        (  # pylint: disable=(unbalanced-tuple-unpacking)
            loss,
            max_pred,
        ) = self.forward(  # pylint: disable=(unbalanced-tuple-unpacking)
            batch, mode="train"
        )
        values = {
            "max_pred": max_pred,
            "train_loss": loss,
        }
        self.log_dict(values, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=(unused-argument)
        (  # pylint: disable=(unbalanced-tuple-unpacking)
            loss,
            max_pred,
        ) = self.forward(  # pylint: disable=(unbalanced-tuple-unpacking)
            batch, mode="val"
        )  # pylint: disable=(unbalanced-tuple-unpacking)
        values = {
            "val_max_pred": max_pred,
            "val_loss": loss,
        }
        self.log_dict(values, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx):  # pylint: disable=(unused-argument)
        (  # pylint: disable=(unbalanced-tuple-unpacking)
            loss,
            max_pred,
        ) = self.forward(  # pylint: disable=(unbalanced-tuple-unpacking)
            batch, mode="test"
        )  # pylint: disable=(unbalanced-tuple-unpacking)
        testues = {
            "test_max_pred": max_pred,
            "test_loss": loss,
        }
        self.log_dict(testues, batch_size=batch.batch_size)

    def predict_step(self, data, batch=None):  # pylint: disable=(unused-argument)
        self.eval()

        with torch.no_grad():
            data = self.undirected_transformer(data)
            if batch is not None:
                x_dict, edge_index_dict, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.batch_dict,
                )
            else:
                x_dict, edge_index_dict, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    None,
                )
            pred_utility = self.model(
                x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
            )

            return pred_utility

    def generate_prob_model(self, explore=False):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        predictions = self.predict_step(data)
        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        return outputs


class RPSChoiceModel(pl.LightningModule):
    def __init__(
        self,
        utility_model,
        lr=0.1,
        weight_decay=0.01,
        warmup=2,
        max_iters=2000,
        **model_kwargs,
    ):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.undirected_transformer = ToUndirected()
        self.example_data = generate_rps_sample()
        self.model = RPSHandChoiceModel(**model_kwargs)
        self.utility_model = utility_model

    def forward(  # pylint: disable=(too-many-locals
        self,
        data,
        mode="train",  # pylint: disable=(unused-argument
    ):
        data = self.undirected_transformer(data)

        x_dict, edge_index_dict, batch_dict, actual_payout = (
            data.x_dict,
            data.edge_index_dict,
            data.batch_dict,
            data.payout,
        )

        predicted_utility = self.utility_model.predict_step(data, batch_dict)

        predicted_regret = self.model(
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )

        payouts_wide, _ = to_dense_batch(
            actual_payout, batch_dict["my_hand"], max_num_nodes=3
        )
        payouts_wide = payouts_wide.reshape(-1, 3)

        immediate_regrets = payouts_wide - predicted_utility

        loss = mse_loss(predicted_regret, target=immediate_regrets)
        max_pred = predicted_regret.max()

        return loss, max_pred

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):  # pylint: disable=(unused-argument)
        (  # pylint: disable=(unbalanced-tuple-unpacking)
            loss,
            max_pred,
        ) = self.forward(  # pylint: disable=(unbalanced-tuple-unpacking)
            batch, mode="train"
        )
        values = {
            "max_pred": max_pred,
            "train_loss": loss,
        }
        self.log_dict(values, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=(unused-argument)
        (  # pylint: disable=(unbalanced-tuple-unpacking)
            loss,
            max_pred,
        ) = self.forward(  # pylint: disable=(unbalanced-tuple-unpacking)
            batch, mode="val"
        )  # pylint: disable=(unbalanced-tuple-unpacking)
        values = {
            "val_max_pred": max_pred,
            "val_loss": loss,
        }
        self.log_dict(values, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx):  # pylint: disable=(unused-argument)
        (  # pylint: disable=(unbalanced-tuple-unpacking)
            loss,
            max_pred,
        ) = self.forward(  # pylint: disable=(unbalanced-tuple-unpacking)
            batch, mode="test"
        )  # pylint: disable=(unbalanced-tuple-unpacking)
        testues = {
            "test_max_pred": max_pred,
            "test_loss": loss,
        }
        self.log_dict(testues, batch_size=batch.batch_size)

    def predict_step(self, data, batch=None):  # pylint: disable=(unused-argument)
        self.eval()

        with torch.no_grad():
            data = self.undirected_transformer(data)
            if batch is not None:
                x_dict, edge_index_dict, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.batch_dict,
                )
            else:
                x_dict, edge_index_dict, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    None,
                )
            pred_regrets = self.model(
                x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
            )

            pred_regrets = torch.maximum(pred_regrets, torch.zeros_like(pred_regrets))
            for i in range(pred_regrets.size(0)):
                row_sum = pred_regrets[i].sum()
                if row_sum == 0:
                    pred_regrets[i] = torch.full_like(pred_regrets[i], 1 / 3)
                else:
                    pred_regrets[i] = pred_regrets[i] / row_sum

            return pred_regrets

    def generate_prob_model(self, explore=False):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        predictions = self.predict_step(data)
        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        return outputs


class RPSRandomWWeightModel:
    def __init__(self, eta):
        self.eta = eta
        self.weights = {"Paper": 1, "Rock": 1, "Scissors": 1}
        self.update_iterations = 0

    def train(self, dataset):
        for data in dataset:
            self.update_weights(data)

    def update_weights(self, data):
        losses = (
            (data.payout.max() - data.payout) / (data.payout.max() - data.payout).max()
        ).reshape(-1)
        new_weights = {}
        for action, loss in zip(data["my_hand"]["choices"], losses):
            update_factor = math.pow((1 - self.eta), float(loss))
            new_weights[action] = self.weights[action] * update_factor
        self.weights = new_weights
        self.update_iterations += 1

        if (self.update_iterations % 1000) == 0:
            self.normalise_weights()

    def normalise_weights(self):
        total_weight = sum(value for _, value in self.weights.items())
        scale_factor = 3 / total_weight
        for key, value in self.weights.items():
            self.weights[key] *= scale_factor

    def generate_prob_model(self, explore=False):
        total_weight = sum(value for _, value in self.weights.items())
        prob_model = [(key, value / total_weight) for key, value in self.weights.items()]
        sorted(prob_model, key=lambda x: x[0])
        return prob_model


class RPSRegretModel:
    def __init__(self, barrier=0.1):
        self.barrier = barrier
        self.prob_model = [("Paper", 1 / 3), ("Rock", 1 / 3), ("Scissors", 1 / 3)]
        self.raw_regrets = None
        self.adjusted_regrets = None

    def train(self, dataset):
        batch = Batch.from_data_list([data for data in dataset])
        choices = [item for sublist in batch["my_hand"].choices for item in sublist]
        regrets = (
            (batch.payout.reshape((-1, 3)) - batch.my_utility.reshape(-1, 1))
            .reshape(-1)
            .numpy()
        )
        exp_regrets = (
            pd.DataFrame({"choices": choices, "regret": regrets})
            .assign(regret=lambda x: np.where(x.regret <= 0, 0, x.regret))
            .groupby("choices")
            .agg(sum_regret=("regret", "sum"))
            .to_dict()["sum_regret"]
        )
        self.raw_regrets = exp_regrets
        adjusted_regrets = {
            key: max(value + self.barrier, 0) for key, value in exp_regrets.items()
        }
        self.adjusted_regrets = adjusted_regrets
        total_regret = sum(value for _, value in adjusted_regrets.items())
        if total_regret <= 0:
            prob_model = [("Paper", 1 / 3), ("Rock", 1 / 3), ("Scissors", 1 / 3)]
        else:
            prob_model = [
                (key, value / total_regret) for key, value in adjusted_regrets.items()
            ]
        self.prob_model = prob_model

    def generate_prob_model(self, explore=False):
        return self.prob_model


class RPSWolfModel:
    def __init__(self, alpha=0.1, delta_win=0.1, delta_lose=0.3, explore_rate=0.1):
        self.alpha = alpha
        self.delta_win = delta_win
        self.delta_lose = delta_lose
        self.q = {"Rock": 0, "Paper": 0, "Scissors": 0}
        self.pi = {"Rock": 1 / 3, "Paper": 1 / 3, "Scissors": 1 / 3}
        self.start_model = {"Rock": 1 / 3, "Paper": 1 / 3, "Scissors": 1 / 3}
        self.observed_policy_count = {"Rock": 1, "Paper": 1, "Scissors": 1}
        self.average_policy = {"Rock": 1 / 3, "Paper": 1 / 3, "Scissors": 1 / 3}
        self.update_iterations = 0
        self.explore_rate = explore_rate

    def train(self, dataset):
        for data in dataset:
            self.update(data)

    def update(self, data):
        taken_action = data.self_action
        for action, reward in zip(data["my_hand"].choices, data.payout):
            self.q[action] = (1 - self.alpha) * self.q[action] + self.alpha * float(
                reward
            )

        self.update_iterations += 1
        self.update_average_policy(taken_action)

        expected_pi_value = sum(self.q[key] * self.pi[key] for key in self.pi)
        expected_avg_value = sum(
            self.q[key] * self.average_policy[key] for key in self.average_policy
        )
        delta = (
            self.delta_win
            if (expected_pi_value >= expected_avg_value)
            else self.delta_lose
        )

        max_action = max(
            {
                action: float(payout)
                for action, payout in zip(data["my_hand"].choices, data.payout)
            },
            key=lambda k: self.q[k],
        )

        for key in self.pi:
            if key == max_action:
                update_value = delta
            else:
                update_value = -delta / 2

            new_value = clip_value(self.pi[key] + update_value)
            self.pi[key] = new_value

        self.normalise_policy(self.pi)

    def update_average_policy(self, taken_action):
        self.observed_policy_count[taken_action] += 1
        total_weight = sum(val for val in self.observed_policy_count.values())
        self.average_policy = {
            key: count_value / total_weight
            for key, count_value in self.observed_policy_count.items()
        }

    def normalise_policy(self, policy):
        total_weight = sum(value for _, value in policy.items())
        scale_factor = 1 / total_weight
        for key, value in policy.items():
            policy[key] *= scale_factor

    def generate_prob_model(self, explore=False):
        if (random.random() < self.explore_rate) and explore:
            # Sometimes we explore
            prob_model = [(key, value) for key, value in self.start_model.items()]
        else:
            prob_model = [(key, value) for key, value in self.pi.items()]
        sorted(prob_model, key=lambda x: x[0])
        return prob_model


if __name__ == "__main__":
    utility_model = RPSSuccessModel(
        hidden_dim=3, final_dim=2, num_layers=1, num_heads=1, dropout=0
    )
    batch = Batch.from_data_list([generate_rps_sample() for _ in range(10)])

    dataset_size = 1000
    dataset = CardDataset([generate_rps_sample() for _ in range(dataset_size)])

    train_set = dataset[: math.floor(dataset_size * 0.7)]
    val_set = dataset[math.floor(dataset_size * 0.7) : math.floor(dataset_size * 0.8)]
    test_set = dataset[math.floor(dataset_size * 0.8) :]

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model_name = "rps_model"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    trainer = pl.Trainer(
        default_root_dir=paths.model_artifact_path / model_name,
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min"),
        ],
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=10,
        enable_progress_bar=False,
        gradient_clip_val=1,
        fast_dev_run=False,
    )

    trainer.fit(utility_model, train_loader, val_loader)

    test_batch = Batch.from_data_list([data for data in test_set])
    utility_model.predict_step(test_batch, test_batch.batch_dict)

    model = RPSChoiceModel(
        utility_model=utility_model,
        hidden_dim=8,
        num_layers=3,
        final_dim=5,
        num_heads=2,
        dropout=0.1,
        input_bias=True,
        bias=True,
        lr=0.1,
        weight_decay=0.01,
    )

    dataset_size = 1000
    dataset = CardDataset([generate_rps_sample() for _ in range(dataset_size)])

    train_set = dataset[: math.floor(dataset_size * 0.7)]
    val_set = dataset[math.floor(dataset_size * 0.7) : math.floor(dataset_size * 0.8)]
    test_set = dataset[math.floor(dataset_size * 0.8) :]

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model_name = "rps_model"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    trainer = pl.Trainer(
        default_root_dir=paths.model_artifact_path / model_name,
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min"),
        ],
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=10,
        enable_progress_bar=False,
        gradient_clip_val=1,
        fast_dev_run=False,
    )

    trainer.fit(model, train_loader, val_loader)

    test_batch = Batch.from_data_list([data for data in test_set])
    model.predict_step(test_batch)
