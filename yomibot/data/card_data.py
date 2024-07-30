import torch
from torch_geometric.data import HeteroData
from torch.nn import Embedding
from yomibot.common.paths import data_path, embeddings_path


class CustomEmbedding:
    allowed_values = None
    storage_path = None
    embedding_dim = None
    
    def __init__(self,string_to_index=string_to_index, embedding=embedding):
        self.string_to_index=string_to_index
        self.embedding = embedding
        
    def __len__(self):
        return len(self.string_to_index)
    
    def __repr__(self):
        return f'{self.__name__}({len(self)}, {self.embedding.embedding_dim})'
    
    @classmethod
    def load(cls):
        obj = cls()
        # embedding_dict = torch.load(embeddings_path / 'yomideck_embedding.pth')
        embedding_dict = torch.load(cls.storage_path)
        obj.string_to_index = embedding_dict['string_to_index']
        embedding = Embedding(embedding_dict['num_cards'], embedding_dict['embedding_dim'])
        embedding.load_state_dict(embedding_dict['embedding'])
        embedding.weight.requires_grad = False
        obj.embedding = embedding
        return obj
        
    @classmethod
    def initialise_embedding(cls, save = False):
        num_cards = len(cls.allowed_values)
        embedding = Embedding(num_cards, cls.embedding_dim)
        embedding.weight.requires_grad = False
        string_to_index = {card_name:int_value for int_value, card_name in enumerate(cls.allowed_values)}
        embedding_dict = {
            'embedding': embedding.state_dict(),
            'num_cards': num_cards,
            'embedding_dim': cls.embedding_dim,
            'string_to_index':string_to_index
        }
        if save:
            cls.storage_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding_dict, cls.storage_path)
        obj = cls(embedding=embedding, string_to_index=string_to_index)    
        return obj
        
    def encode(self, card_name_iterable):
        card_list = list(card_name_iterable)
        long_list = [self.string_to_index[card] for card in card_list]
        input_tensor = torch.tensor(long_list, dtype=torch.long)
        encoding = self.embedding(input_tensor)
        return encoding


class YomiEmbedding(CustomEmbedding):
    allowed_values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'LB','HB', 'Dodge', 'Thr', 'X','Y', 'Z', 'Abi1', 'Abi2', 'Burst', 'Super1', 'Super2', 'unknown']
    storage_path = embeddings_path / 'yomideck_embedding.pth'
    embedding_dim = 8

class YomiPlayerEmbedding(CustomEmbedding):
    allowed_values = ['Grave', 'Jaina', 'Argagarg', 'Geiger','BBB', 'Setsuki', 'Valerie', 'Rook', 'Midori', 'Bigby', 'Troq',
                        'Onimaru', 'Vendetta', 'River', 'Lum', 'Degrey', 'Menelker', 'Persephone']
    storage_path = embeddings_path / 'yomiplayer_embedding.pth'
    embedding_dim = 8

class GemEmbedding(CustomEmbedding):
    allowed_values = ['Red', 'Green', 'Blue', 'Black', 'White', 'Purple','Orange', 'Diamond']
    storage_path = embeddings_path / 'gem_embedding.pth'
    embedding_dim = 5


def generate_sample_data():
    
    yomi_encoder = YomiEmbedding.load()
    player_encoder = YomiPlayerEmbedding.load()
    gem_encoder = GemEmbedding.load()

    my_hand = ['LB','HB', 'Thr', 'Burst', 'A', 'B', 'E']
    my_discard = ['Super1', 'Super2']
    opponent_hand = ['LB','HB', 'Thr', 'Burst', 'unknown', 'unknown', 'unknown']
    opponent_discard = ['Super1', 'Super2']
    my_player = ['Grave']
    opponent_player = ['Jaina']
    my_gem = ['Red']
    opponent_gem = ['Blue']
    
    yomi_data = HeteroData()
    yomi_data['my_hand'].x = yomiencoder.encode(my_hand)
    yomi_data['my_discard'].x = yomiencoder.encode(my_discard)
    yomi_data['opponent_hand'].x = yomiencoder.encode(opponent_hand)
    yomi_data['opponent_discard'].x = yomiencoder.encode(opponent_discard)
    yomi_data.my_health = torch.tensor(100)
    yomi_data.opponent_health = torch.tensor(100)
    yomi_data.my_player = player_encoder.encode(my_player)
    yomi_data.opponent_player = player_encoder.encode(opponent_player)
    yomi_data.my_gem = gem_encoder.encode(my_gem)
    yomi_data.opponent_gem = gem_encoder.encode(opponent_gem)
    return yomi_data