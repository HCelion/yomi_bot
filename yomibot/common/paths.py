from pathlib import Path

root_path = Path(__file__).parents[2]

data_path = root_path / 'data'
yomi_path = root_path / 'yomibot'
embeddings_path = yomi_path / 'data' / 'embeddings'
