import math
from copy import deepcopy
import pandas as pd
import logging
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
from pytorch_lightning.loggers import TensorBoardLogger
from yomibot.graph_models.base_encoder import (
    RPSHandChoiceModel,
    RPSUtilityModel,
    RPSPolicyActorModel,
)
from yomibot.graph_models.helpers import (
    CosineWarmupScheduler,
    MetricsCallback,
    plot_model_history,
    plot_model_history_ternary,
    parse_log_item,
    plot_model_history_with_mse,
    parse_freq_log_item,
    CircularBuffer,
    parse_q_log_item,
    checkpoint_callback,
    empirical_frequencies,
)
from yomibot.data.card_data import generate_rps_sample, CardDataset
from yomibot.common import paths
from yomibot.data.card_data import (
    rps_non_standard_payout,
    rps_standard_payout,
    rps_non_standard_payout_opponent,
)
import warnings

warnings.filterwarnings("ignore")

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)


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

        return {
            "optimizer": optimizer,
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


class RPSAvgActionModel(pl.LightningModule):
    def __init__(
        self,
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
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(  # pylint: disable=(too-many-locals
        self,
        data,
        mode="train",  # pylint: disable=(unused-argument
    ):
        data = self.undirected_transformer(data)

        x_dict, edge_index_dict, batch_dict = (
            data.x_dict,
            data.edge_index_dict,
            data.batch_dict,
        )

        action_mapping = {"Rock": 0, "Paper": 1, "Scissors": 2}
        chosen_actions = torch.tensor(
            [action_mapping[val] for val in data.self_action]
        ).reshape(-1, 1)
        mapped_list_of_lists = torch.tensor(
            [
                [action_mapping[item] for item in sublist]
                for sublist in data["my_hand"].choices
            ]
        ).reshape(-1, 3)

        best_actions = (chosen_actions == mapped_list_of_lists).float()
        predicted_densities = self.model(
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )
        predictions = softmax(predicted_densities)
        loss = self.loss_func(predicted_densities, target=best_actions)
        max_pred = predictions.max()

        return loss, max_pred

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return {
            "optimizer": optimizer,
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
            predicted_densities = self.model(
                x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
            )
            predictions = torch.exp(predicted_densities) / (
                torch.exp(predicted_densities).sum(axis=1).reshape(-1, 1)
            )

            return predictions

    def generate_prob_model(self, explore=False, as_dict=False):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        predictions = self.predict_step(data)
        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        if as_dict:
            return {key: value for key, value in outputs}
        return outputs


class RPSChoiceModel(pl.LightningModule):
    def __init__(
        self,
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
        predicted_value = self.model(
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )

        payouts_wide, _ = to_dense_batch(
            actual_payout, batch_dict["my_hand"], max_num_nodes=3
        )
        payouts_wide = payouts_wide.reshape(-1, 3)
        loss = mse_loss(predicted_value, target=payouts_wide)
        max_pred = predicted_value.max()

        return loss, max_pred

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return {
            "optimizer": optimizer,
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

            return pred_regrets

    def generate_value(self, as_dict=False):
        data = generate_rps_sample()
        data = self.undirected_transformer(data)
        choices = data["my_hand"].choices
        x_dict, edge_index_dict = (
            data.x_dict,
            data.edge_index_dict,
        )
        predictions = self.model(x_dict, edge_index_dict)
        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        if as_dict:
            return {key: value for key, value in outputs}
        return outputs


class RPSCurrentActionModel(pl.LightningModule):
    def __init__(
        self,
        avg_policy_model,
        lr=0.1,
        weight_decay=0.01,
        max_iters=2000,
        delta_win=0.001,
        delta_lose=0.002,
        **model_kwargs,
    ):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.undirected_transformer = ToUndirected()
        self.example_data = generate_rps_sample()
        self.model = RPSHandChoiceModel(**model_kwargs)
        self.avg_policy_model = avg_policy_model
        self.delta_win = delta_win
        self.delta_lose = delta_lose

    def forward(  # pylint: disable=(too-many-locals
        self,
        data,
        mode="train",  # pylint: disable=(unused-argument
    ):
        data = self.undirected_transformer(data)
        with torch.no_grad():
            avg_policies = self.avg_policy_model.predict_step(data, batch=data.batch_dict)
            current_policy_frozen = self.predict_step(data, batch=data.batch_dict)

        x_dict, edge_index_dict, batch_dict = (
            data.x_dict,
            data.edge_index_dict,
            data.batch_dict,
        )

        current_policy_logit = self.model(
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )

        counterfactual_payouts = data.payout.reshape(-1, 3)

        historic_exp_reward = (
            (counterfactual_payouts * avg_policies).sum(axis=1).reshape(-1, 1)
        )
        current_exp_reward = (
            (counterfactual_payouts * current_policy_frozen).sum(axis=1).reshape(-1, 1)
        )

        weights = torch.where(
            (historic_exp_reward - current_exp_reward) > 0,
            self.delta_lose,
            self.delta_win,
        )

        loss = (
            (softmax(current_policy_logit, dim=1) * counterfactual_payouts).sum(axis=1)
            * weights
        ).mean()
        max_pred = current_policy_frozen.max()

        return loss, max_pred

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return {
            "optimizer": optimizer,
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
            predicted_densities = self.model(
                x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
            )
            predictions = torch.exp(predicted_densities) / (
                torch.exp(predicted_densities).sum(axis=1).reshape(-1, 1)
            )

            return predictions

    def generate_prob_model(self, explore=False, as_dict=False):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        predictions = self.predict_step(data)
        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        if as_dict:
            return {key: value for key, value in outputs}
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


class ClonedActor(pl.LightningModule):
    def __init__(self, cloned_model, lr=0.01, weight_decay=0.00001):
        super(ClonedActor, self).__init__()
        self.actor_net = cloned_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.undirected_transformer = ToUndirected()

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        policy, _, _ = self.actor_net(
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )
        return policy

    def predict_step(self, data, batch=None):
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

            preds, _, _ = self.actor_net(
                x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
            )

        return preds

    def generate_prob_model(self, as_dict=False):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        predictions = self.predict_step(data)
        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        if as_dict:
            return {key: value for key, value in outputs}
        return outputs

    def training_step(self, batch, batch_idx):
        x_dict, edge_index_dict, batch_dict = (
            batch.x_dict,
            batch.edge_index_dict,
            batch.batch_dict,
        )
        policy = self.forward(x_dict, edge_index_dict, batch_dict)
        targets = batch.action_index
        log_probs = torch.log(policy)
        loss = (batch.my_utility * (-log_probs * targets).sum(dim=1)).mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class ClonedFrequencyActor(ClonedActor):
    def training_step(self, batch, batch_idx):
        x_dict, edge_index_dict, batch_dict = (
            batch.x_dict,
            batch.edge_index_dict,
            batch.batch_dict,
        )
        policy = self.forward(x_dict, edge_index_dict, batch_dict)
        targets = batch.action_index
        counterfactual_outcomes = batch.payout.reshape(-1, 3)
        # log_probs = torch.log(policy)
        counterfactual_outcomes * policy
        # loss = -(log_probs*targets).sum(dim=1).mean()
        loss = -(counterfactual_outcomes * policy).sum(dim=1).mean()
        return loss


class ActorCritic(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=4,
        num_layers=1,
        lr=0.01,
        M=100,
        anticipatory_parameter=0.1,
        payout_functions=[rps_standard_payout, rps_standard_payout],
        weight_decay=0.00001,
        N_win=50,
        N_loss=100,
        circ_buffer_size=1000,
        epsilon=0.1,
        update_epochs=20,
        frequency_epochs=100,
    ):
        super(ActorCritic, self).__init__()
        self.actor_nets = [
            RPSPolicyActorModel(hidden_dim=hidden_dim, num_layers=num_layers)
            for _ in range(2)
        ]
        self.frequency_nets = [
            RPSPolicyActorModel(hidden_dim=hidden_dim, num_layers=num_layers)
            for _ in range(2)
        ]
        self.value_nets = [
            RPSPolicyActorModel(hidden_dim=hidden_dim, num_layers=num_layers)
            for _ in range(2)
        ]
        self.lr = lr
        self.weight_decay = weight_decay
        self.standard_probs = [("Rock", 1 / 3), ("Paper", 1 / 3), ("Scissors", 1 / 3)]
        self.M = M
        self.N_win = N_win
        self.N_loss = N_loss
        self.undirected_transformer = ToUndirected()
        self.update_epochs = update_epochs
        self.payout_functions = payout_functions
        self.frequency_epochs = frequency_epochs
        self.automatic_optimization = False
        self.lt_memory = [CircularBuffer(size=circ_buffer_size) for _ in range(2)]

    def forward(self, batch, actor_index=0):
        if isinstance(batch, Batch):
            x_dict, edge_index_dict, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                batch.batch_dict,
            )
        else:
            x_dict, edge_index_dict, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                None,
            )
        policy, _, _ = self.q_nets[actor_index](
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )
        return policy

    def frequency_forward(self, batch, actor_index=0):
        if isinstance(batch, Batch):
            x_dict, edge_index_dict, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                batch.batch_dict,
            )
        else:
            x_dict, edge_index_dict, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                None,
            )
        frequency, _, _ = self.frequency_nets[actor_index](
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )
        return frequency

    def predict_step(self, data, batch=None, actor_index=0, frequency=False):
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

            if frequency:
                preds, _, _ = self.frequency_nets[actor_index](
                    x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
                )
            else:
                preds, _, _ = self.actor_nets[actor_index](
                    x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
                )

        return preds

    def q_values(self, actor_index=0):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        predictions = self.predict_step(data, actor_index=actor_index, frequency=False)
        predictions = predictions.reshape(1, 3)

        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        return {key: value for key, value in outputs}

    def valuation(self, batch, actor_index=0):
        if isinstance(batch, Batch):
            x_dict, edge_index_dict, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                batch.batch_dict,
            )
        else:
            x_dict, edge_index_dict, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                None,
            )
        _, values, _ = self.value_nets[actor_index](
            x_dict=x_dict, edge_index_dict=edge_index_dict, batch_dict=batch_dict
        )
        return values

    def configure_optimizers(self):
        actor_optimizers = [
            optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for net in self.actor_nets
        ]
        critic_optimizers = [
            optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for net in self.value_nets
        ]
        frequency_optimizers = [
            optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for net in self.frequency_nets
        ]
        return actor_optimizers + critic_optimizers + frequency_optimizers

    def generate_prob_model(self, as_dict=False, actor_index=0, frequency=False):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        if frequency:
            predictions = self.predict_step(data, actor_index=actor_index, frequency=True)

        else:
            predictions = self.predict_step(
                data, actor_index=actor_index, frequency=False
            )
            predictions = predictions.reshape(1, 3)

        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        if as_dict:
            return {key: value for key, value in outputs}
        return outputs

    def collect_trajectory(
        self, num_iterations, self_model, opponent_model, actor_index=0, record=False
    ):
        states, actions, rewards = [], [], []
        opponent_index = 1 if actor_index == 0 else 0
        for _ in range(num_iterations):
            state = generate_rps_sample(
                payout_function=self.payout_functions[actor_index],
                self_model=self_model,
                opponent_model=opponent_model,
                opp_payout_function=self.payout_functions[opponent_index],
                mirror=False,
            )

            action = state.self_action
            action_index = state["my_hand"]["choices"].index(state.self_action)
            states.append(state)
            actions.append(action)
            rewards.append(state.my_utility)
        return states, actions, rewards

    def compute_returns(self, rewards):
        return torch.FloatTensor(rewards)

    def policy_update(self, policy_model, states):
        model_updated = ClonedActor(
            deepcopy(policy_model), weight_decay=self.weight_decay, lr=self.lr
        )
        dataloader = DataLoader(states, batch_size=256, shuffle=True)
        trainer = pl.Trainer(
            max_epochs=self.update_epochs,
            logger=False,
            enable_progress_bar=False,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model_updated, dataloader)
        return model_updated

    def frequency_update(self, policy_model, states):
        model_updated = ClonedFrequencyActor(
            deepcopy(policy_model), weight_decay=self.weight_decay, lr=self.lr
        )
        dataloader = DataLoader(states, batch_size=256, shuffle=True)
        trainer = pl.Trainer(
            max_epochs=self.frequency_epochs,
            logger=False,
            enable_progress_bar=False,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model_updated, dataloader)
        return model_updated

    def training_step(self, batch, batch_idx):
        current_epoch = self.current_epoch
        if current_epoch != 0:
            self_model = self.generate_prob_model(actor_index=0)
            other_model = self.generate_prob_model(actor_index=1)
        else:
            self_model = self.standard_probs
            other_model = self.standard_probs

        states1, actions1, rewards1 = self.collect_trajectory(
            self_model=self_model,
            opponent_model=other_model,
            record=True,
            num_iterations=self.M,
        )
        states2, actions2, rewards2 = self.collect_trajectory(
            self_model=other_model,
            opponent_model=self_model,
            record=True,
            num_iterations=self.M,
        )

        self.lt_memory[0].add_set(states1)
        self.lt_memory[1].add_set(states2)

        self.frequency_nets[0] = self.frequency_update(
            self.frequency_nets[0], states1
        ).actor_net
        self.frequency_nets[1] = self.frequency_update(
            self.frequency_nets[1], states2
        ).actor_net

        empirical_frequencies_1 = empirical_frequencies(actions1)
        empirical_frequencies_2 = empirical_frequencies(actions2)

        mean_model1 = self.generate_prob_model(actor_index=0, frequency=True)
        mean_model2 = self.generate_prob_model(actor_index=1, frequency=True)
        _, mean_actions1, rewards1_mean = self.collect_trajectory(
            self_model=mean_model1,
            opponent_model=other_model,
            record=False,
            num_iterations=self.M,
        )
        _, mean_actions2, rewards2_mean = self.collect_trajectory(
            self_model=mean_model2,
            opponent_model=self_model,
            record=False,
            num_iterations=self.M,
        )

        utility1 = np.mean(rewards1)
        utility2 = np.mean(rewards2)
        mean_utility1 = np.mean(rewards1_mean)
        mean_utility2 = np.mean(rewards2_mean)

        model1_is_winning = 1 if (utility1 > mean_utility1) else 0
        model2_is_winning = 1 if (utility2 > mean_utility2) else 0

        N1 = self.N_win if (utility1 > mean_utility1) else self.N_loss
        N2 = self.N_win if (utility2 > mean_utility2) else self.N_loss

        # empirical_frequencies([s.opponent_action for s in states1])
        model1_updated = self.policy_update(self.actor_nets[0], states1)
        model2_updated = self.policy_update(self.actor_nets[1], states2)

        updated_model1_probs = model1_updated.generate_prob_model()
        updated_model2_probs = model2_updated.generate_prob_model()
        states_update1, updated_actions1, _ = self.collect_trajectory(
            self_model=self_model,
            opponent_model=other_model,
            record=True,
            num_iterations=N1,
        )
        states_update2, updated_actions2, _ = self.collect_trajectory(
            self_model=self_model,
            opponent_model=other_model,
            record=True,
            num_iterations=N2,
        )

        self.actor_nets[0] = self.frequency_update(
            self.actor_nets[0], states1 + states_update1
        ).actor_net
        self.actor_nets[1] = self.frequency_update(
            self.actor_nets[1], states2 + states_update2
        ).actor_net

        empirical_updated_probs1 = empirical_frequencies(actions1 + updated_actions1)
        empirical_updated_probs2 = empirical_frequencies(actions2 + updated_actions2)

        model1 = self.generate_prob_model(actor_index=0, as_dict=True)
        model2 = self.generate_prob_model(actor_index=1, as_dict=True)
        m1_metrics = {"model_1_" + key: value for key, value in model1.items()}
        m2_metrics = {"model_2_" + key: value for key, value in model2.items()}
        emp_m1_metrics = {
            "model_1_emp_" + key: value for key, value in empirical_frequencies_1.items()
        }
        emp_m2_metrics = {
            "model_2_emp_" + key: value for key, value in empirical_frequencies_2.items()
        }
        updated_model1_probs = model1_updated.generate_prob_model(as_dict=True)
        updated_model2_probs = model2_updated.generate_prob_model(as_dict=True)
        upd_model_1_metrics = {
            "upd_model_1_" + key: value for key, value in updated_model1_probs.items()
        }
        upd_model_2_metrics = {
            "upd_model_2_" + key: value for key, value in updated_model2_probs.items()
        }
        emp_upd_model_1_metrics = {
            "upd_model_1_emp_" + key: value
            for key, value in empirical_updated_probs1.items()
        }
        emp_upd_model_2_metrics = {
            "upd_model_2_emp_" + key: value
            for key, value in empirical_updated_probs2.items()
        }

        freq_model1 = self.generate_prob_model(
            actor_index=0, as_dict=True, frequency=True
        )
        freq_model2 = self.generate_prob_model(
            actor_index=1, as_dict=True, frequency=True
        )
        freq_m1_metrics = {
            "freq_model_1_" + key: value for key, value in freq_model1.items()
        }
        freq_m2_metrics = {
            "freq_model_2_" + key: value for key, value in freq_model2.items()
        }

        combined_metrics = {
            **m1_metrics,
            **m2_metrics,
            **freq_m1_metrics,
            **freq_m2_metrics,
            **upd_model_1_metrics,
            **upd_model_2_metrics,
            **emp_upd_model_1_metrics,
            **emp_upd_model_2_metrics,
            "model1_is_winning": model1_is_winning,
            "model2_is_winning": model2_is_winning,
            **emp_m1_metrics,
            **emp_m2_metrics,
        }
        combined_metrics["epoch"] = current_epoch

        self.log_dict(combined_metrics)


# from yomibot.graph_models.helpers import get_nash_equilibria
# # A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
# A = np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]])
# player_1_optimum, player_2_optimum = get_nash_equilibria(A)
#
# non_standard_situation = [rps_non_standard_payout, rps_non_standard_payout_opponent]
# standard_situation = [rps_standard_payout, rps_standard_payout]
# actor_critic = ActorCritic(M=1000, payout_functions = non_standard_situation, weight_decay=0.0001, lr=0.001,
#                           N_win = 10, N_loss =20, frequency_epochs=100, update_epochs=1, circ_buffer_size=100000)
# self=actor_critic
#
#
# metrics_callback = MetricsCallback()
# logger = TensorBoardLogger(save_dir=paths.log_paths, name='rps_logs')
#
# #TODO add counts
# # Implement Matching Pennies for better analysis
# trainer = pl.Trainer(max_epochs=200,logger=logger,  callbacks=[metrics_callback])
# trainer.fit(actor_critic, [0])
#
# logged_metrics = metrics_callback.metrics
# float_dicts = []
# for dictionary in logged_metrics:
#     float_dicts.append( {key:val.item() for key, val in dictionary.items()})
#
# pd.DataFrame(float_dicts)[['model1_is_winning', 'model2_is_winning']].cov()
#
# model1_history = [parse_log_item(log,1) for log in logged_metrics]
# model2_history = [parse_log_item(log, 2) for log in logged_metrics]
# freq_model1_history = [parse_freq_log_item(log,1) for log in logged_metrics]
# freq_model2_history = [parse_freq_log_item(log,2) for log in logged_metrics]
#
#
# model1_history[-5:]
#
# model2_history[-5:]
#
# fig = plot_model_history(model1_history, player_1_optimum)
# fig = plot_model_history(model2_history, player_2_optimum)
# fig = plot_model_history(freq_model1_history, player_1_optimum)
# fig = plot_model_history(freq_model2_history, player_2_optimum)
#
#
# fig = plot_model_history_ternary(model1_history, player_1_optimum)
# fig = plot_model_history_ternary(model2_history, player_2_optimum)
#
# fig = plot_model_history_with_mse(model1_history, player_1_optimum)
# fig = plot_model_history_with_mse(model2_history, player_2_optimum)
