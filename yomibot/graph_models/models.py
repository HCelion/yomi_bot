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
    checkpoint_callback,
    empirical_frequencies,
    get_empirical_ratios,
    get_empirical_regrets,
    turn_to_penny_df,
    plot_penny,
    PennyReservoir,
    invert_payout_dictionary
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
        return loss


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

    def generate_q_values(self):
        prob_dict = {}
        for state in [1, 2, 3]:
            data = generate_penny_sample(state=state)
            choices = data["my_hand"].choices
            with torch.no_grad():
                data = self.undirected_transformer(data)
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
            regrets = {choice:round(float(val),3) for choice, val in zip(choices, q_values)}
            prob_dict[state] = regrets
        return prob_dict

    def max_q_diff(self, states):
        empirical_regrets = get_empirical_regrets(states)
        q_values = flatten_dict(self.generate_q_values())
        max_diff = 0
        for key, value in empirical_regrets.items():
            max_diff = max(max_diff, abs(value -q_values[key] ))
        return round(max_diff,3)

    def max_sign_diff(self, states):
        empirical_regrets = get_empirical_regrets(states)
        q_values = flatten_dict(self.generate_q_values())
        max_signdiff = 0
        for key, value in empirical_regrets.items():
            max_signdiff = max(max_signdiff,  int(np.sign(value) != np.sign(q_values[key])) )
        return max_signdiff

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


        loss = ((regret_predictions - regrets).square().sum(dim=1)).mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer




class PennyRegretMinimiser(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=4,
        num_layers=2,
        lr=0.01,
        M=100,
        max_diff = 0.01,
        payout_dictionary=None,
        weight_decay=0.00001,
        frequency_epochs=100,
        update_epochs=100,
        max_iterations=10,
        starting_probs1=None,
        starting_probs2=None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        super(PennyRegretMinimiser, self).__init__()
        self.actor_nets = [
            RegretActor(
                PennyPolicyActorModel(hidden_dim=hidden_dim, num_layers=num_layers)
            )
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
        self.max_diff = max_diff
        self.max_iterations = max_iterations
        self.M = M
        self.undirected_transformer = ToUndirected()
        self.payout_dictionary = payout_dictionary
        self.frequency_epochs = frequency_epochs
        self.automatic_optimization = False
        self.update_epochs = update_epochs
        self.lt_memory = [
            PennyReservoir("first_reservoir",payout_dictionary = payout_dictionary),
            PennyReservoir("second_reservoir",payout_dictionary = invert_payout_dictionary(payout_dictionary))
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

    def collect_trajectory(self, self_model, opponent_model, epoch=None):
        states1, actions1, rewards1 = [], [], []
        states2, actions2, rewards2 = [], [], []

        for state in [1,2,3]:
            state1, state2 = generate_penny_sample(
                payout_dictionary=self.payout_dictionary,
                self_model=self_model,
                opponent_model=opponent_model,
                state=state,
                mirror=True,
            )

            action = state1.self_action
            other_action = state2.self_action
            states1.append(state1)
            states2.append(state2)
            actions1.append(action)
            actions2.append(other_action)
            rewards1.append(state1.my_utility)
            rewards2.append(state2.my_utility)

        if epoch is not None:
            for state in states1:
                state.weight = 1
            for state in states2:
                state.weight =  1

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
        model_updated.actor_net._initialize_policy_head()
        dataloader = DataLoader(states, batch_size=3000, shuffle=True)

        max_its = 0
        while  (model_updated.max_sign_diff(states) > 0) and (model_updated.max_q_diff(states) >  self.max_diff) and (max_its < self.max_iterations):
            trainer = pl.Trainer(
                max_epochs=self.update_epochs,
                logger=False,
                enable_progress_bar=False,
                callbacks=[checkpoint_callback],
                gradient_clip_val=0.5,
            )
            trainer.fit(model_updated, dataloader)
            max_its += 1

        max_its = 0
        while  (model_updated.max_sign_diff(states) > 0) and (max_its < self.max_iterations):
            print('In second iteration')
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
        dataloader = DataLoader(states, batch_size=256, shuffle=True)

        trainer = pl.Trainer(
            max_epochs=self.frequency_epochs,
            logger=False,
            enable_progress_bar=False,
            callbacks=[checkpoint_callback],
            # gradient_clip_val=0.5,
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
            self_model = self.starting_probs1
            other_model = self.starting_probs2
        else:
            self_model = self.generate_prob_model(actor_index=0)
            other_model = self.generate_prob_model(actor_index=1)


        m1_metrics = flatten_dict(self_model, "model_1")
        m2_metrics = flatten_dict(other_model, "model_2")

        states1, actions1, rewards1, states2, actions2, rewards2 = (
            self.collect_trajectory(
                self_model=self_model,
                opponent_model=other_model,
                epoch=current_epoch,
            )
        )

        self.lt_memory[0].store_data(states1)
        self.lt_memory[1].store_data(states2)

        state_sample1 = self.lt_memory[0].sample(self.M)
        state_sample2 = self.lt_memory[1].sample(self.M)

        # get_empirical_ratios(state_sample1,'test')
        # get_empirical_ratios(state_sample2,'test')
        # get_empirical_regrets(state_sample1)
        # get_empirical_regrets(state_sample2)
        self.actor_nets[0] = self.policy_update(
            self.actor_nets[0].actor_net, state_sample1
        )
        self.actor_nets[1] = self.policy_update(
            self.actor_nets[1].actor_net, state_sample2
        )


        # get_empirical_regrets(state_sample1)
        self.actor_nets[1].generate_q_values()
        # self.actor_nets[1].max_sign_diff(state_sample2)
        # get_empirical_regrets(state_sample2)

        emp_alpha = flatten_dict(get_empirical_ratios(states1, 'alpha'), 'play_hist1')
        emp_beta = flatten_dict(get_empirical_ratios(states2, 'beta'), 'play_hist2')
        emp_regret1 = flatten_dict(get_empirical_regrets(states1), 'emp_regret_1')
        emp_regret2 = flatten_dict(get_empirical_regrets(states2), 'emp_regret_2')
        model_1_q_values = flatten_dict(self.actor_nets[0].generate_q_values(), 'q_values_1')
        model_2_q_values = flatten_dict(self.actor_nets[1].generate_q_values(), 'q_values_2')

        combined_metrics = {
            **m1_metrics,
            **m2_metrics,
            **emp_alpha,
            **emp_beta,
            **emp_regret1,
            **emp_regret2,
            **model_1_q_values,
            **model_2_q_values
        }
        combined_metrics["epoch"] = current_epoch

        self.log_dict(combined_metrics)


from yomibot.graph_models.helpers import get_nash_equilibria

# A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
# A = np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]])
# A = np.array([[1, -1], [-1, 1]])
# A = np.array([[1, -1], [-1, 1]])
# A = np.array([[4, -1], [-1, 1]])

A = np.array([[-4, 1], [1, -1]])

player_1_optimum, player_2_optimum = get_nash_equilibria(A)


non_standard_situation = [rps_non_standard_payout, rps_non_standard_payout_opponent]
standard_situation = [rps_standard_payout, rps_standard_payout]

# payout_dictionary = {1:(penny_standard_payout,penny_opponent_standard_payout), 2:(penny_standard_payout,penny_opponent_standard_payout),
#                     3:(penny_standard_payout,penny_opponent_standard_payout)}

# payout_dictionary = {1:(penny_non_standard_payout,penny_non_standard_payout_opponent), 2:(penny_non_standard_payout,penny_non_standard_payout_opponent),
#                     3:(penny_non_standard_payout,penny_non_standard_payout_opponent)}


#
#
payout_dictionary = {
    1: (penny_standard_payout, penny_opponent_standard_payout),
    2: (penny_non_standard_payout, penny_non_standard_payout_opponent),
    3: (penny_non_standard_payout_opponent, penny_non_standard_payout),
}

actor_critic = PennyRegretMinimiser(
    M=1000,
    payout_dictionary=payout_dictionary,
    weight_decay=0.00001,
    lr=0.2,
    update_epochs=500,
    max_diff = 0.2,
    max_iterations = 10,
    frequency_epochs=100,
    starting_probs1={"Odd": 0.8, "Even": 0.2},
    starting_probs2={"Odd": 0.8, "Even": 0.2},
)

self = actor_critic

metrics_callback = MetricsCallback()
logger = TensorBoardLogger(save_dir=paths.log_paths, name="penny_logs")

trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=[metrics_callback])
trainer.fit(actor_critic, [0])


logged_metrics = metrics_callback.metrics

penny_df = turn_to_penny_df(logged_metrics)



(penny_df
    .pivot(index = ['state','epoch'], columns= 'policy', values = 'mean_policy')
    .reset_index()
    .query('state == 2')
    .plot( x='alpha', y='beta')
    # .tail()
)
