from yomibot.data.card_data import generate_sample_data, generate_small_sample_data
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, CGConv
from torch_geometric.transforms import ToUndirected
import torch_geometric.nn as geom_nn
from torch.nn.functional import sigmoid, softmax
from torch_geometric.utils import to_dense_batch

class HandConv(torch.nn.Module):
    
    def __init__(self, num_layers, hidden_dim, num_heads, reduction_method='sum', bias=True, dropout=False, batch_norm=False, layer_norm=False):
        super(HandConv, self).__init__()
        self.hetero_layers = nn.ModuleList()
        self.nodewise_layers = nn.ModuleList()
        
        for i in range(num_layers):
            hetero_conv = HeteroConv({
                ('my_hand', 'beats', 'opponent_hand'): GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
                ('my_hand', 'loses_to', 'opponent_hand'): CGConv(channels=(hidden_dim, hidden_dim)),
                ('opponent_hand', 'rev_beats', 'my_hand'): CGConv(channels=(hidden_dim, hidden_dim)),
                ('opponent_hand', 'rev_loses_to', 'my_hand'): GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False)
            }, aggr=reduction_method)
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


class YomiSuccessModel(nn.Module):
    
    def __init__(self, hidden_dim,final_dim, input_bias=True, **kwargs):
        super(YomiSuccessModel, self).__init__()
        sample_data = generate_sample_data()
        card_embed_dim = sample_data['my_hand'].x.shape[-1]

        # Initial encoding
        self.linear_encoder = nn.Linear(card_embed_dim, hidden_dim, bias=input_bias)
        self.hand_endcoder = HandConv(hidden_dim=hidden_dim, **kwargs)
        self.final_encoder = nn.Sequential(
            nn.Linear(2*hidden_dim, final_dim,bias=input_bias), 
            nn.LeakyReLU(final_dim, inplace=True),
            nn.Linear(final_dim, 1,bias=input_bias), 
            nn.LeakyReLU(1, inplace=True),
            )
        
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        x_dict = {key: self.linear_encoder(x) for key, x in x_dict.items()}
        x_dict = self.hand_endcoder(x_dict, edge_index_dict)
        if batch_dict:
            my_hand_encoding = geom_nn.global_mean_pool(x_dict['my_hand'], batch=batch_dict['my_hand'])
            other_hand_encoding = geom_nn.global_mean_pool(x_dict['opponent_hand'], batch=batch_dict['opponent_hand'])
        else:
            my_hand_encoding = geom_nn.global_mean_pool(x_dict['my_hand'])
            other_hand_encoding = geom_nn.global_mean_pool(x_dict['opponent_hand'])
        full_encoding = torch.cat((my_hand_encoding, other_hand_encoding), dim=1)
        logits = self.final_encoder(full_encoding)
        return logits


class YomiHandChoiceModel(nn.Module):
    def __init__(self, hidden_dim,final_dim, input_bias=True, **kwargs):
        super(YomiHandChoiceModel, self).__init__()
        sample_data = generate_sample_data()
        card_embed_dim = sample_data['my_hand'].x.shape[-1]

        # Initial encoding
        self.linear_encoder = nn.Linear(card_embed_dim, hidden_dim, bias=input_bias)
        self.hand_endcoder = HandConv(hidden_dim=hidden_dim, **kwargs)
        self.final_encoder = nn.Sequential(
            nn.Linear(hidden_dim, final_dim,bias=input_bias), 
            nn.LeakyReLU(final_dim, inplace=True),
            nn.Linear(final_dim, 1,bias=input_bias), 
            )
        
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        x_dict = {key: self.linear_encoder(x) for key, x in x_dict.items()}
        x_dict = self.hand_endcoder(x_dict, edge_index_dict)
        my_hand =  x_dict['my_hand']
        logits = self.final_encoder(my_hand)
        output_tensor, indices =  to_dense_batch(logits, batch_dict['my_hand'], max_num_nodes=12, fill_value=-9999)
        output_logits = output_tensor.reshape(-1,12)
        return output_logits
        
