from yomibot.data.card_data import (
    generate_sample_data,
    generate_rps_sample,
    generate_penny_sample,
    rps_standard_payout,
    penny_standard_payout,
    penny_opponent_standard_payout,
)
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, CGConv
import torch_geometric.nn as geom_nn
from torch_geometric.utils import to_dense_batch


class HandConv(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        reduction_method="mean",
        bias=True,
        dropout=False,
        batch_norm=False,
        layer_norm=False,
    ):
        super(HandConv, self).__init__()
        self.hetero_layers = nn.ModuleList()
        self.nodewise_layers = nn.ModuleList()

        for i in range(num_layers):
            hetero_conv = HeteroConv(
                {
                    ("my_hand", "beats", "opponent_hand"): GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        concat=False,
                        add_self_loops=False,
                    ),
                    ("my_hand", "loses_to", "opponent_hand"): CGConv(
                        channels=(hidden_dim, hidden_dim)
                    ),
                    ("opponent_hand", "rev_beats", "my_hand"): CGConv(
                        channels=(hidden_dim, hidden_dim)
                    ),
                    ("opponent_hand", "rev_loses_to", "my_hand"): GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        concat=False,
                        add_self_loops=False,
                    ),
                },
                aggr=reduction_method,
            )
            self.hetero_layers.append(hetero_conv)

            nodewise_layer = nn.ModuleList()
            nodewise_layer.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if batch_norm:
                nodewise_layer.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                nodewise_layer.append(nn.LayerNorm(hidden_dim))
            if dropout:
                nodewise_layer.append(nn.Dropout(dropout))
            self.nodewise_layers.append(nodewise_layer)

    def forward(self, x_dict, edge_index_dict):
        for hetero_conv, nodewise_layer in zip(self.hetero_layers, self.nodewise_layers):
            x_dict = hetero_conv(x_dict, edge_index_dict)
            for key in x_dict.keys():
                for layer in nodewise_layer:
                    x_dict[key] = layer(x_dict[key])

        return x_dict


class SimpleHandConv(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        bias=True,
        dropout=False,
        batch_norm=False,
        layer_norm=False,
    ):
        super(SimpleHandConv, self).__init__()
        self.nodewise_layers = nn.ModuleList()

        for i in range(num_layers):
            nodewise_layer = nn.ModuleList()
            nodewise_layer.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if batch_norm:
                nodewise_layer.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                nodewise_layer.append(nn.LayerNorm(hidden_dim))
            if dropout:
                nodewise_layer.append(nn.Dropout(dropout))
            self.nodewise_layers.append(nodewise_layer)

    def forward(self, x_dict, edge_index_dict):
        for nodewise_layer in self.nodewise_layers:
            for key in x_dict.keys():
                for layer in nodewise_layer:
                    x_dict[key] = layer(x_dict[key])

        return x_dict


class YomiSuccessModel(nn.Module):
    def __init__(self, hidden_dim, final_dim, input_bias=True, **kwargs):
        super(YomiSuccessModel, self).__init__()
        sample_data = generate_sample_data()
        card_embed_dim = sample_data["my_hand"].x.shape[-1]

        # Initial encoding
        self.linear_encoder = nn.Linear(card_embed_dim, hidden_dim, bias=input_bias)
        self.hand_endcoder = HandConv(hidden_dim=hidden_dim, **kwargs)
        self.final_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, final_dim, bias=input_bias),
            nn.LeakyReLU(final_dim, inplace=True),
            nn.Linear(final_dim, 1, bias=input_bias),
            nn.LeakyReLU(1, inplace=True),
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        x_dict = {key: self.linear_encoder(x) for key, x in x_dict.items()}
        x_dict = self.hand_endcoder(x_dict, edge_index_dict)
        if batch_dict:
            my_hand_encoding = geom_nn.global_mean_pool(
                x_dict["my_hand"], batch=batch_dict["my_hand"]
            )
            other_hand_encoding = geom_nn.global_mean_pool(
                x_dict["opponent_hand"], batch=batch_dict["opponent_hand"]
            )
        else:
            my_hand_encoding = geom_nn.global_mean_pool(x_dict["my_hand"], batch=None)
            other_hand_encoding = geom_nn.global_mean_pool(
                x_dict["opponent_hand"], batch=None
            )
        full_encoding = torch.cat((my_hand_encoding, other_hand_encoding), dim=1)
        logits = self.final_encoder(full_encoding)
        return logits


class YomiHandChoiceModel(nn.Module):
    def __init__(self, hidden_dim, final_dim, input_bias=True, **kwargs):
        super(YomiHandChoiceModel, self).__init__()
        sample_data = generate_sample_data()
        card_embed_dim = sample_data["my_hand"].x.shape[-1]

        # Initial encoding
        self.linear_encoder = nn.Linear(card_embed_dim, hidden_dim, bias=input_bias)
        self.hand_endcoder = HandConv(hidden_dim=hidden_dim, **kwargs)
        self.final_encoder = nn.Sequential(
            nn.Linear(hidden_dim, final_dim, bias=input_bias),
            nn.LeakyReLU(final_dim, inplace=True),
            nn.Linear(final_dim, 1, bias=input_bias),
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        x_dict = {key: self.linear_encoder(x) for key, x in x_dict.items()}
        x_dict = self.hand_endcoder(x_dict, edge_index_dict)
        my_hand = x_dict["my_hand"]
        logits = self.final_encoder(my_hand)

        if batch_dict is not None:
            output_tensor, _ = to_dense_batch(
                logits, batch_dict["my_hand"], max_num_nodes=12, fill_value=-9999
            )
        else:
            # we just pad here to have consistent output
            output_tensor = torch.concatenate(
                [logits, torch.full((12 - len(logits), 1), -9999)]
            )

        output_logits = output_tensor.reshape(-1, 12)
        return output_logits


class RPSHandChoiceModel(nn.Module):
    def __init__(self, hidden_dim, final_dim, input_bias=True, **kwargs):
        super(RPSHandChoiceModel, self).__init__()
        sample_data = generate_rps_sample()
        card_embed_dim = sample_data["my_hand"].x.shape[-1]

        # Initial encoding
        self.linear_encoder = nn.Linear(card_embed_dim, hidden_dim, bias=input_bias)
        self.hand_endcoder = SimpleHandConv(hidden_dim=hidden_dim, **kwargs)
        self.final_encoder = nn.Sequential(
            nn.Linear(hidden_dim, final_dim, bias=input_bias),
            nn.LeakyReLU(final_dim, inplace=True),
            nn.Linear(final_dim, 1, bias=input_bias),
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        x_dict = {key: self.linear_encoder(x) for key, x in x_dict.items()}
        x_dict = self.hand_endcoder(x_dict, edge_index_dict)
        my_hand = x_dict["my_hand"]
        logits = self.final_encoder(my_hand)
        if batch_dict is not None:
            output_tensor, _ = to_dense_batch(
                logits, batch_dict["my_hand"], max_num_nodes=3, fill_value=-9999
            )
        else:
            output_tensor = logits
        output_logits = output_tensor.reshape(-1, 3)
        return output_logits


class PennyPolicyActorModel(nn.Module):
    def __init__(self, hidden_dim, input_bias=True, **kwargs):
        super(PennyPolicyActorModel, self).__init__()
        sample_data = generate_penny_sample()
        self.hand_size = 2

        card_embed_dim = sample_data["my_hand"].x.shape[-1]

        # Initial encoding
        self.linear_encoder = nn.Linear(card_embed_dim, hidden_dim, bias=input_bias)
        self.hand_endcoder = SimpleHandConv(hidden_dim=hidden_dim, **kwargs)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=input_bias),
            nn.LeakyReLU(1, inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=input_bias),
        )

        self.q_value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=input_bias),
            nn.LeakyReLU(2, inplace=True),
        )

        self.softmax = nn.Softmax(dim=1)

        self._initialize_policy_head()

    def _initialize_policy_head(self):
        # Initialize weights to zero and biases to zero
        nn.init.constant_(self.policy_head[0].weight, 0)
        if self.policy_head[0].bias is not None:
            nn.init.constant_(self.policy_head[0].bias, 0)

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        x_dict = {key: self.linear_encoder(x) for key, x in x_dict.items()}
        x_dict = self.hand_endcoder(x_dict, edge_index_dict)
        policy = self.policy_head(x_dict["my_hand"])

        if batch_dict is not None:
            policy, _ = to_dense_batch(
                policy, batch_dict["my_hand"], max_num_nodes=2, fill_value=-9999
            )
            policy = policy.reshape((-1, 2))
            policy = self.softmax(policy)
            my_hand_encoding = geom_nn.global_mean_pool(
                x_dict["my_hand"], batch=batch_dict["my_hand"]
            )
        else:
            policy = policy.T
            policy = self.softmax(policy)
            my_hand_encoding = geom_nn.global_mean_pool(x_dict["my_hand"], batch=None)

        value = self.value_head(my_hand_encoding)
        q_values = self.q_value_head(x_dict["my_hand"])
        return policy, value, q_values


class RPSPolicyActorModel(nn.Module):
    def __init__(self, hidden_dim, input_bias=True, **kwargs):
        super(RPSPolicyActorModel, self).__init__()
        sample_data = generate_rps_sample()
        self.hand_size = 3

        card_embed_dim = sample_data["my_hand"].x.shape[-1]

        # Initial encoding
        self.linear_encoder = nn.Linear(card_embed_dim, hidden_dim, bias=input_bias)
        self.hand_endcoder = SimpleHandConv(hidden_dim=hidden_dim, **kwargs)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=input_bias),
            nn.LeakyReLU(1, inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=input_bias),
        )

        self.q_value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=input_bias),
            nn.LeakyReLU(3, inplace=True),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        x_dict = {key: self.linear_encoder(x) for key, x in x_dict.items()}
        x_dict = self.hand_endcoder(x_dict, edge_index_dict)
        policy = self.policy_head(x_dict["my_hand"])

        if batch_dict is not None:
            policy, _ = to_dense_batch(
                policy, batch_dict["my_hand"], max_num_nodes=3, fill_value=-9999
            )
            policy = policy.reshape((-1, 3))
            policy = self.softmax(policy)
            my_hand_encoding = geom_nn.global_mean_pool(
                x_dict["my_hand"], batch=batch_dict["my_hand"]
            )
        else:
            policy = policy.T
            policy = self.softmax(policy)
            my_hand_encoding = geom_nn.global_mean_pool(x_dict["my_hand"], batch=None)

        value = self.value_head(my_hand_encoding)
        q_values = self.q_value_head(x_dict["my_hand"])
        return policy, value, q_values
