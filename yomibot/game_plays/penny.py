import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from yomibot.graph_models.models import PennyRegretMinimiser
from yomibot.graph_models.helpers import get_nash_equilibria
from yomibot.data.card_data import (
    penny_standard_payout,
    penny_non_standard_payout,
    penny_opponent_standard_payout,
    penny_non_standard_payout_opponent,
)
from yomibot.graph_models.helpers import (
    MetricsCallback,
    turn_to_penny_df,
    plot_penny,
)
from yomibot.common import paths


if __name__ == "__main__":
    non_standard_A = np.array([[-4, 1], [1, -1]])
    player_1_optimum_non_standard, player_2_optimum_non_standard = get_nash_equilibria(
        non_standard_A
    )
    standard_A = np.array([[1, -1], [-1, 1]])
    player_1_optimum_standard, player_2_optimum_standard = get_nash_equilibria(standard_A)
    optima = {
        1: player_1_optimum_standard,
        2: player_1_optimum_non_standard,
        3: player_1_optimum_non_standard,
    }

    # payout_dictionary = {1:(penny_standard_payout,penny_opponent_standard_payout), 2:(penny_standard_payout,penny_opponent_standard_payout),
    #                     3:(penny_standard_payout,penny_opponent_standard_payout)}

    # payout_dictionary = {1:(penny_non_standard_payout,penny_non_standard_payout_opponent), 2:(penny_non_standard_payout,penny_non_standard_payout_opponent),
    #                     3:(penny_non_standard_payout,penny_non_standard_payout_opponent)}

    payout_dictionary = {
        1: (penny_standard_payout, penny_opponent_standard_payout),
        2: (penny_non_standard_payout, penny_non_standard_payout_opponent),
        3: (penny_non_standard_payout_opponent, penny_non_standard_payout),
    }

    actor_critic = PennyRegretMinimiser(
        M=500,
        payout_dictionary=payout_dictionary,
        weight_decay=0.00001,
        lr=0.2,
        update_epochs=500,
        max_diff=0.2,
        max_iterations=5,
        frequency_epochs=100,
        starting_probs1={"Odd": 0.8, "Even": 0.2},
        starting_probs2={"Odd": 0.8, "Even": 0.2},
    )

    metrics_callback = MetricsCallback()
    logger = TensorBoardLogger(save_dir=paths.log_paths, name="penny_logs")

    trainer = pl.Trainer(max_epochs=600, logger=logger, callbacks=[metrics_callback])
    trainer.fit(actor_critic, [0])

    logged_metrics = metrics_callback.metrics
    penny_df = turn_to_penny_df(logged_metrics)
    penny_fig = plot_penny(penny_df, optima)
