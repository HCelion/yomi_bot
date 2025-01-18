from pathlib import Path

root_path = Path(__file__).parents[2]

data_path = root_path / "data"
model_artifact_path = root_path / "model"
yomi_path = root_path / "yomibot"
embeddings_path = yomi_path / "data" / "embeddings"
log_paths = root_path / "logs"
