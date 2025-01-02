import math
from time import time
from copy import deepcopy
import pandas as pd
import logging
import numpy as np
import random
from torch_geometric.data import Batch
from torch.nn.functional import (
    mse_loss,
    softmax,
    sigmoid,
    binary_cross_entropy_with_logits,
)
import torch.nn as nn
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
    RPSPolicyActorModel,
    PennyPolicyActorModel,
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
    turn_to_penny_df,
    plot_penny,
    PennyReservoir,
)
from yomibot.data.helpers import flatten_dict, unflatten_dict
from yomibot.data.card_data import generate_rps_sample, generate_penny_sample, CardDataset
from yomibot.common import paths
from yomibot.data.card_data import (
    rps_non_standard_payout,
    rps_standard_payout,
    rps_non_standard_payout_opponent,
    penny_standard_payout,
    penny_opponent_standard_payout,
    penny_non_standard_payout,
    penny_non_standard_payout_opponent,
    penny_even_payout,
    penny_odd_payout,
    get_penny_regrets,
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
    def __init__(self, cloned_model, lr=0.01, weight_decay=0.00001, type="rps"):
        super(ClonedActor, self).__init__()
        self.actor_net = cloned_model
        self.type = type
        self.hand_size = self.actor_net.hand_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.undirected_transformer = ToUndirected()

    def forward(self, x_dict, edge_index_dict, state_x, batch_dict=None):
        policy, _, _ = self.actor_net(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            state_x=state_x,
        )
        return policy

    def predict_step(self, data, batch=None):
        with torch.no_grad():
            data = self.undirected_transformer(data)
            if batch is not None:
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    data.batch_dict,
                )
            else:
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    None,
                )

            preds, _, _ = self.actor_net(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                state_x=state_x,
            )

        return preds

    def generate_prob_model(self):
        prob_dict = {}
        for state in [1, 2, 3]:
            if self.type == "rps":
                data = generate_rps_sample()
            else:
                data = generate_penny_sample(state=state)
            choices = data["my_hand"].choices
            predictions = self.predict_step(data)
            outputs = [
                (category, float(value))
                for category, value in zip(choices, predictions[0])
            ]
            outputs = sorted(outputs, key=lambda x: x[0])
            prob_dict[state] = {key: value for key, value in outputs}
        return prob_dict

    def get_overfit(self):
        all_max_predictions = []
        prob_model = self.generate_prob_model()
        for state in [1, 2, 3]:
            max_prediction = max(val for _, val in prob_model[state].items())
            all_max_predictions.append(max_prediction)
        return min(all_max_predictions)

    def training_step(self, batch, batch_idx):
        x_dict, edge_index_dict, state_x, batch_dict = (
            batch.x_dict,
            batch.edge_index_dict,
            batch.state_x,
            batch.batch_dict,
        )
        policy = self.forward(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            state_x=state_x,
            batch_dict=batch_dict,
        )
        targets = batch.action_index

        log_probs = torch.log(policy)
        loss = (batch.my_utility * (-log_probs * targets).sum(dim=1)).mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class RegretActor(pl.LightningModule):
    def __init__(self, cloned_model, lr=0.01, weight_decay=0.00001, type="rps"):
        super(RegretActor, self).__init__()
        self.actor_net = cloned_model
        self.type = type
        self.hand_size = self.actor_net.hand_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.undirected_transformer = ToUndirected()

    def forward(self, x_dict, edge_index_dict, state_x, batch_dict=None):
        _, _, q_values = self.actor_net(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            state_x=state_x,
        )
        return q_values

    def predict_step(self, data, batch=None):
        with torch.no_grad():
            data = self.undirected_transformer(data)
            if batch is not None:
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    data.batch_dict,
                )
            else:
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    None,
                )
            _, _, q_values = self.actor_net(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                state_x=state_x,
            )
            q_values = q_values.reshape(-1, 2)
            positive_policy = torch.clamp(q_values, min=0)
            sum_per_row = positive_policy.sum(dim=1, keepdim=True)
            weights = positive_policy / sum_per_row
            nan_mask = torch.isnan(weights)
            weights = torch.where(nan_mask, torch.tensor(0.0), weights)
            zero_sum_mask = (sum_per_row == 0).float()
            preds = weights + zero_sum_mask * 0.5

        return preds

    def generate_prob_model(self):
        prob_dict = {}
        for state in [1, 2, 3]:
            data = generate_penny_sample(state=state)
            choices = data["my_hand"].choices
            predictions = self.predict_step(data)
            outputs = [
                (category, float(value))
                for category, value in zip(choices, predictions[0])
            ]
            outputs = sorted(outputs, key=lambda x: x[0])
            prob_dict[state] = {key: value for key, value in outputs}
        return prob_dict

    def get_overfit(self):
        all_max_predictions = []
        prob_model = self.generate_prob_model()
        for state in [1, 2, 3]:
            max_prediction = max(val for _, val in prob_model[state].items())
            all_max_predictions.append(max_prediction)
        return min(all_max_predictions)

    def training_step(self, batch, batch_idx):
        x_dict, edge_index_dict, state_x, batch_dict = (
            batch.x_dict,
            batch.edge_index_dict,
            batch.state_x,
            batch.batch_dict,
        )
        regret_predictions = self.forward(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            state_x=state_x,
            batch_dict=batch_dict,
        )
        regrets = batch.regret

        # regrets_wide, _ = to_dense_batch(
        #     regrets, batch_dict["my_hand"], max_num_nodes=2
        # )
        # regrets_wide = regrets_wide.reshape(-1,2)
        #

        loss = ((regret_predictions - regrets).square().sum(dim=1)).mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class ClonedFrequencyActor(ClonedActor):
    def predict_step(self, data, batch=None):
        with torch.no_grad():
            data = self.undirected_transformer(data)
            if batch is not None:
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    data.batch_dict,
                )
            else:
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    None,
                )

            # state_x1 = generate_penny_sample(state=1).state_x
            # state_x2 = generate_penny_sample(state=2).state_x
            # state_x3 = generate_penny_sample(state=3).state_x

            _, _, preds = self.actor_net(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                batch_dict=batch_dict,
                state_x=state_x,
            )

            preds = torch.sigmoid(preds)

        return preds

    def forward(self, x_dict, edge_index_dict, state_x, batch_dict=None):
        _, _, q_values = self.actor_net(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            state_x=state_x,
        )
        return q_values

    def generate_prob_model(self):
        prob_dict = {}
        for state in [1, 2, 3]:
            data = generate_penny_sample(state=state)
            choices = data["my_hand"].choices
            predictions = self.predict_step(data)
            predictions = predictions / (predictions).sum()
            predictions = predictions.reshape(-1, 2)
            outputs = [
                (category, float(value))
                for category, value in zip(choices, predictions[0])
            ]
            outputs = sorted(outputs, key=lambda x: x[0])
            prob_dict[state] = {key: value for key, value in outputs}
        return prob_dict

    def training_step(self, batch, batch_idx):
        x_dict, edge_index_dict, state_x, batch_dict = (
            batch.x_dict,
            batch.edge_index_dict,
            batch.state_x,
            batch.batch_dict,
        )
        logits = self.forward(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            state_x=state_x,
            batch_dict=batch_dict,
        )

        logits, _ = to_dense_batch(logits, batch_dict["my_hand"], max_num_nodes=2)
        logits = logits.reshape(-1, 2)
        targets = batch.action_index.float()
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        #
        # if hasattr(batch, "weight"):
        #     weights = batch.weight / batch.weight.sum()
        #     loss = (loss*weights.reshape(-1,1)).sum()
        # else:
        #     loss = loss.sum(dim=1).mean()
        return loss


class PennyActorCritic(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=4,
        num_layers=2,
        lr=0.01,
        M=100,
        payout_dictionary=None,
        weight_decay=0.00001,
        N_win=50,
        N_loss=100,
        circ_buffer_size=1000,
        epsilon=0.1,
        update_epochs=20,
        frequency_epochs=100,
        starting_probs1=None,
        starting_probs2=None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        super(PennyActorCritic, self).__init__()
        self.actor_nets = [
            PennyPolicyActorModel(hidden_dim=hidden_dim, num_layers=num_layers)
            for _ in range(2)
        ]
        self.frequency_nets = [
            ClonedFrequencyActor(
                PennyPolicyActorModel(hidden_dim=hidden_dim, num_layers=num_layers)
            )
            for _ in range(2)
        ]

        self.lr = lr
        self.weight_decay = weight_decay
        self.M = M
        self.N_win = N_win
        self.N_loss = N_loss
        self.undirected_transformer = ToUndirected()
        self.update_epochs = update_epochs
        self.payout_dictionary = payout_dictionary
        self.frequency_epochs = frequency_epochs
        self.automatic_optimization = False
        self.lt_memory = [
            PennyReservoir("player_" + str(i), payout_dictionary=payout_dictionary)
            for i in range(2)
        ]
        self.starting_probs1 = {state: starting_probs1 for state in [1, 2, 3]}
        self.starting_probs2 = {state: starting_probs2 for state in [1, 2, 3]}

    def forward(self, batch, actor_index=0):
        if isinstance(batch, Batch):
            x_dict, edge_index_dict, state_x, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                batch.state_x,
                batch.batch_dict,
            )
        else:
            x_dict, edge_index_dict, state_x, batch_dict = (
                batch.x_dict,
                batch.edge_index_dict,
                batch.state_x,
                None,
            )
        policy, _, _ = self.actor_nets[actor_index](
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
            state_x=state_x,
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
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    data.batch_dict,
                )
            else:
                x_dict, edge_index_dict, state_x, batch_dict = (
                    data.x_dict,
                    data.edge_index_dict,
                    data.state_x,
                    None,
                )

            if frequency:
                state_x1 = generate_penny_sample(state=1).state_x
                state_x2 = generate_penny_sample(state=2).state_x
                state_x3 = generate_penny_sample(state=3).state_x

                preds, _, _ = self.frequency_nets[actor_index](
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    batch_dict=batch_dict,
                    state_x=state_x,
                )
                preds, _, _ = self.frequency_nets[actor_index](
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    batch_dict=batch_dict,
                    state_x=state_x3,
                )

            else:
                preds, _, _ = self.actor_nets[actor_index](
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    batch_dict=batch_dict,
                    state_x=state_x,
                )

        return preds

    def generate_prob_model(self, actor_index=0, frequency=False):
        if frequency:
            prob_dict = self.frequency_nets[actor_index].generate_prob_model()
        else:
            prob_dict = self.actor_nets[actor_index].generate_prob_model()

        return prob_dict

    def collect_trajectory(self, num_iterations, self_model, opponent_model, epoch=None):
        states1, actions1, rewards1 = [], [], []
        states2, actions2, rewards2 = [], [], []

        for _ in range(num_iterations):
            state, other_state = generate_penny_sample(
                payout_dictionary=self.payout_dictionary,
                self_model=self_model,
                opponent_model=opponent_model,
                mirror=True,
            )
            action = state.self_action
            other_action = other_state.self_action
            states1.append(state)
            states2.append(other_state)
            actions1.append(action)
            actions2.append(other_action)
            rewards1.append(state.my_utility)
            rewards2.append(other_state.my_utility)

        if epoch is not None:
            for state in states1:
                state.weight = epoch + 1
            for state in states2:
                state.weight = epoch + 1

        return states1, actions1, rewards1, states2, actions2, rewards2

    def compute_returns(self, rewards):
        return torch.FloatTensor(rewards)

    def policy_update(self, policy_model, states):
        model_updated = RegretActor(
            deepcopy(policy_model),
            weight_decay=self.weight_decay,
            lr=self.lr,
            type="penny",
        )
        dataloader = DataLoader(states, batch_size=256, shuffle=True)

        trainer = pl.Trainer(
            max_epochs=self.update_epochs,
            logger=False,
            enable_progress_bar=False,
            callbacks=[checkpoint_callback],
            gradient_clip_val=0.5,
        )
        trainer.fit(model_updated, dataloader)
        return model_updated

    def frequency_update(self, states):
        model_updated = ClonedFrequencyActor(
            PennyPolicyActorModel(hidden_dim=self.hidden_dim, num_layers=self.num_layers),
            weight_decay=self.weight_decay,
            lr=self.lr,
            type="penny",
        )
        # self = model_updated
        dataloader = DataLoader(states, batch_size=256, shuffle=True)

        trainer = pl.Trainer(
            max_epochs=self.frequency_epochs,
            logger=False,
            enable_progress_bar=False,
            callbacks=[checkpoint_callback],
            gradient_clip_val=0.5,
        )
        trainer.fit(model_updated, dataloader)

        return model_updated

    def configure_optimizers(self):
        actor_optimizers = [
            optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for net in self.actor_nets
        ]

        frequency_optimizers = [
            optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for net in self.frequency_nets
        ]
        return actor_optimizers + frequency_optimizers

    def training_step(self, batch, batch_idx):
        current_epoch = self.current_epoch

        if (current_epoch == 0) and self.starting_probs1:
            for reservoir in self.lt_memory:
                reservoir.clear_reservoir()
            start_states1, _, _, start_states2, _, _ = self.collect_trajectory(
                self_model=self.starting_probs1,
                opponent_model=self.starting_probs2,
                num_iterations=self.M,
            )

            self.actor_nets[0] = self.frequency_update(start_states1)
            self.actor_nets[1] = self.frequency_update(start_states2)
            self_model = self.generate_prob_model(actor_index=0)
            other_model = self.generate_prob_model(actor_index=1)
            old_model1 = self.generate_prob_model(actor_index=0)
            old_model2 = self.generate_prob_model(actor_index=1)
        else:
            self.actor_nets[0] = self.policy_update(
                self.actor_nets[0].actor_net, self.lt_memory[0].sample(self.M)
            )
            self.actor_nets[1] = self.policy_update(
                self.actor_nets[1].actor_net, self.lt_memory[1].sample(self.M)
            )
            self_model = self.generate_prob_model(actor_index=0)
            other_model = self.generate_prob_model(actor_index=1)
            old_model1 = self.generate_prob_model(actor_index=0)
            old_model2 = self.generate_prob_model(actor_index=1)

        self_model = {
            1: {"Even": 0, "Odd": 1},
            2: {"Even": 1, "Odd": 0},
            3: {"Even": 0, "Odd": 1},
        }
        # Clean up ClonedFrequencyActor
        states1, actions1, rewards1, states2, actions2, rewards2 = (
            self.collect_trajectory(
                self_model=self_model,
                opponent_model=other_model,
                num_iterations=self.M,
                epoch=current_epoch,
            )
        )

        # batch = Batch.from_data_list(states1)

        self.lt_memory[0].store_data(states1)
        self.lt_memory[1].store_data(states2)

        self.frequency_nets[0] = self.frequency_update(self.lt_memory[0].sample(self.M))

        self.frequency_nets[1] = self.frequency_update(self.lt_memory[1].sample(self.M))

        model1 = self.generate_prob_model(actor_index=0)
        model2 = self.generate_prob_model(actor_index=1)
        freq1 = self.generate_prob_model(actor_index=0, frequency=True)
        freq2 = self.generate_prob_model(actor_index=1, frequency=True)

        m1_metrics = flatten_dict(model1, "model_1")
        m2_metrics = flatten_dict(model2, "model_2")
        freq_m1_metrics = flatten_dict(freq1, "freq_model_1")
        freq_m2_metrics = flatten_dict(freq2, "freq_model_2")

        combined_metrics = {
            **m1_metrics,
            **m2_metrics,
            **freq_m1_metrics,
            **freq_m2_metrics,
        }
        combined_metrics["epoch"] = current_epoch

        self.log_dict(combined_metrics)


from yomibot.graph_models.helpers import get_nash_equilibria

# A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
# A = np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]])
# A = np.array([[1, -1], [-1, 1]])
# A = np.array([[1, -1], [-1, 1]])
A = np.array([[4, -1], [-1, 1]])
player_1_optimum, player_2_optimum = get_nash_equilibria(A)


non_standard_situation = [rps_non_standard_payout, rps_non_standard_payout_opponent]
standard_situation = [rps_standard_payout, rps_standard_payout]

# payout_dictionary = {1:(penny_standard_payout,penny_opponent_standard_payout), 2:(penny_non_standard_payout,penny_non_standard_payout_opponent),
#                     3:(penny_non_standard_payout_opponent,penny_non_standard_payout)}

# payout_dictionary = {1:(penny_even_payout,penny_odd_payout), 2:(penny_odd_payout,penny_even_payout),
#                     3:(penny_even_payout,penny_odd_payout)}


payout_dictionary = {
    1: (penny_standard_payout, penny_opponent_standard_payout),
    2: (penny_standard_payout, penny_opponent_standard_payout),
    3: (penny_standard_payout, penny_opponent_standard_payout),
}

actor_critic = PennyActorCritic(
    M=1000,
    payout_dictionary=payout_dictionary,
    weight_decay=0.0001,
    lr=0.05,
    N_win=100,
    N_loss=200,
    frequency_epochs=200,
    update_epochs=200,
    circ_buffer_size=100000,
    starting_probs1={"Odd": 0.8, "Even": 0.2},
    starting_probs2={"Odd": 0.2, "Even": 0.8},
)

self = actor_critic


metrics_callback = MetricsCallback()
logger = TensorBoardLogger(save_dir=paths.log_paths, name="penny_logs")

trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=[metrics_callback])
trainer.fit(actor_critic, [0])

logged_metrics = metrics_callback.metrics
logged_metrics[0]

standard_model = {s: {"Odd": 0.5, "Even": 0.5} for s in [1, 2, 3]}
states = [
    generate_penny_sample(
        state=s,
        self_model=standard_model,
        other_model=standard_model,
        payout_dictionary=payout_dictionary,
    )
    for s in [1, 2, 3]
]

states[0]["my_hand"].choices
states[0].regret

# TODO: Make 1 and 2 train on the same game

#
#
# model1_history = [parse_log_item(log,1) for log in logged_metrics]
# model2_history = [parse_log_item(log, 2) for log in logged_metrics]
# freq_model1_history = [parse_freq_log_item(log,1) for log in logged_metrics]
# freq_model2_history = [parse_freq_log_item(log,2) for log in logged_metrics]
#
#
# fig = plot_model_history(model1_history, player_1_optimum)
# fig = plot_model_history(model2_history, player_2_optimum)
# fig = plot_model_history(freq_model1_history, player_1_optimum)
# fig = plot_model_history(freq_model2_history, player_2_optimum)
#


penny_df = turn_to_penny_df(logged_metrics)
penny_df.tail(100)[["m1_winning", "m2_winning"]].corr()

penny_df.tail(20).freq_alpha.mean()
penny_df.tail(20).freq_beta.mean()

(
    penny_df.assign(d_alpha=lambda x: x.alpha - x.old_alpha)
    .assign(d_beta=lambda x: x.beta - x.old_beta)
    .plot(x="beta", y="d_alpha")
)

penny_df.plot(x="epoch", y="alpha_emp_update")
penny_df.plot(x="epoch", y="beta_emp_update")
penny_df.plot(x="epoch", y="beta")
penny_df.plot(x="epoch", y="freq_alpha")
penny_df.plot(x="epoch", y="freq_beta")
penny_df[["m1_winning", "m2_winning"]].corr()


(
    penny_df.assign(
        error=lambda x: np.sqrt(
            (x.freq_alpha - 0.5) * (x.freq_alpha - 0.5)
            + (x.freq_beta - 0.5) * (x.freq_beta - 0.5)
        )
    ).error.plot()
)

(
    penny_df.assign(
        error=lambda x: np.sqrt(
            (x.freq_alpha - 0.5) * (x.freq_alpha - 0.5)
            + (x.freq_beta - 0.5) * (x.freq_beta - 0.5)
        )
    ).error.plot()
)

plot_penny(
    penny_df, add_freq_trace=True, add_emp_update_trace=False, add_update_trace=False
)


# model1_history = [parse_log_item(log,1) for log in logged_metrics]
# model2_history = [parse_log_item(log, 2) for log in logged_metrics]
# freq_model1_history = [parse_freq_log_item(log,1) for log in logged_metrics]
# freq_model2_history = [parse_freq_log_item(log,2) for log in logged_metrics]
#
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
