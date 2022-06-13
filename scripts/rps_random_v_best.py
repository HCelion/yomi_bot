from src.rps.rps_arena import RPSArena
import pandas as pd
from tqdm import tqdm

outcomes = []

for _ in range(100):
    arena = RPSArena(left_strategy="random", right_strategy="random", num_simulations=5)
    outcome = arena.play_game()
    outcomes.append(outcome)

outcome_random_df = pd.DataFrame.from_records(outcomes)

outcome_random_df["score_diff"] = (
    outcome_random_df.left_score - outcome_random_df.right_score
)
outcome_random_df.score_diff.hist()


best_outcomes = []

for _ in tqdm(range(100), desc="Running games ..."):
    arena = RPSArena(left_strategy="random", right_strategy="best", num_simulations=5)
    outcome = arena.play_game()
    best_outcomes.append(outcome)

outcome_best_df = pd.DataFrame.from_records(best_outcomes)
outcome_best_df["score_diff"] = outcome_best_df.left_score - outcome_best_df.right_score


outcome_random_df.score_diff.mean()
outcome_best_df.score_diff.mean()

outcome_best_df.score_diff.hist()
