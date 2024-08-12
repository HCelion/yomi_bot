import torch
from torch_geometric.data import Data, Batch
from yomibot.data.card_data import generate_sample_data, generate_small_sample_data
from yomibot.graph_models.base_encoder import YomiSuccessModel, YomiHandChoiceModel
from torch_geometric.transforms import ToUndirected
from torch.nn.functional import softmax


def generate_mixed_batch():
    batch = Batch.from_data_list(
        [generate_sample_data() for _ in range(6)]
        + [generate_small_sample_data() for _ in range(3)]
    )
    return batch


def test_success_model_batch():
    model = YomiSuccessModel(
        hidden_dim=8,
        num_layers=3,
        final_dim=5,
        num_heads=2,
        dropout=0.2,
        input_bias=True,
        bias=True,
    )
    _ = model.eval()
    batch = generate_mixed_batch()
    undirected_transformer = ToUndirected()
    batch = undirected_transformer(batch)
    log_predictions = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
    assert log_predictions.shape == (len(batch), 1)
    return True


def test_choice_model_batch():
    model = YomiHandChoiceModel(
        hidden_dim=8,
        num_layers=3,
        final_dim=5,
        num_heads=2,
        dropout=0.2,
        input_bias=True,
        bias=True,
    )
    _ = model.eval()
    batch = generate_mixed_batch()
    undirected_transformer = ToUndirected()
    batch = undirected_transformer(batch)
    log_predictions = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
    predictions = softmax(log_predictions, dim=1)
    assert predictions.shape == (len(batch), 12)

    # Prove that non existent cards are assigned zero value
    assert torch.all(predictions[:6, 7:] == 0)
    assert not torch.all(predictions[:6, 6:] == 0)
    assert torch.all(predictions[-3:, 4:] == 0)
    assert not torch.all(predictions[-3:, 3:] == 0)
    return True


def test_success_model_single():
    model = YomiSuccessModel(
        hidden_dim=8,
        num_layers=3,
        final_dim=5,
        num_heads=2,
        dropout=0.2,
        input_bias=True,
        bias=True,
    )
    _ = model.eval()
    data = generate_sample_data()
    undirected_transformer = ToUndirected()
    data = undirected_transformer(data)
    log_predictions = model(data.x_dict, data.edge_index_dict)
    return True


def test_choice_model_single():
    model = YomiHandChoiceModel(
        hidden_dim=8,
        num_layers=3,
        final_dim=5,
        num_heads=2,
        dropout=0.2,
        input_bias=True,
        bias=True,
    )
    _ = model.eval()
    data = generate_sample_data()
    undirected_transformer = ToUndirected()
    data = undirected_transformer(data)
    log_predictions = model(data.x_dict, data.edge_index_dict)
    predictions = softmax(log_predictions, dim=1)

    assert predictions.shape == (1, 12)
    assert torch.all(predictions[:, 7:] == 0)
    assert not torch.all(predictions[:, 6:] == 0)
    return True
