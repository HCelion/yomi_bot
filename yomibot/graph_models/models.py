import math
from torch_geometric.data import Batch
from torch.nn.functional import softmax, mse_loss
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

    def generate_prob_model(self):
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

    def generate_prob_model(self):
        data = generate_rps_sample()
        choices = data["my_hand"].choices
        predictions = self.predict_step(data)
        outputs = [
            (category, float(value)) for category, value in zip(choices, predictions[0])
        ]
        outputs = sorted(outputs, key=lambda x: x[0])
        return outputs


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