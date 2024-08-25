import torch
from random import choice, choices, shuffle
from torch import zeros
from torch_geometric.data import HeteroData, Dataset
from torch.nn import Embedding
from yomibot.common.paths import embeddings_path


class CustomEmbedding:
    allowed_values = None
    storage_path = None
    embedding_dim = None

    def __init__(self, string_to_index, embedding):
        self.string_to_index = string_to_index
        self.embedding = embedding

    def __len__(self):
        return len(self.string_to_index)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)}, {self.embedding.embedding_dim})"

    @classmethod
    def load(cls):
        # embedding_dict = torch.load(embeddings_path / 'yomideck_embedding.pth')
        embedding_dict = torch.load(cls.storage_path)
        embedding = Embedding(
            embedding_dict["num_cards"], embedding_dict["embedding_dim"]
        )
        obj = cls(string_to_index=embedding_dict["string_to_index"], embedding=embedding)
        obj.embedding.load_state_dict(embedding_dict["embedding"])
        obj.embedding.weight.requires_grad = False
        return obj

    @classmethod
    def initialise_embedding(cls, save=False):
        num_cards = len(cls.allowed_values)
        embedding = Embedding(num_cards, cls.embedding_dim)
        embedding.weight.requires_grad = False
        string_to_index = {
            card_name: int_value for int_value, card_name in enumerate(cls.allowed_values)
        }
        embedding_dict = {
            "embedding": embedding.state_dict(),
            "num_cards": num_cards,
            "embedding_dim": cls.embedding_dim,
            "string_to_index": string_to_index,
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
    allowed_values = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "LB",
        "HB",
        "Dodge",
        "Thr",
        "X",
        "Y",
        "Z",
        "Abi1",
        "Abi2",
        "Burst",
        "Super1",
        "Super2",
        "unknown",
    ]
    storage_path = embeddings_path / "yomideck_embedding.pth"
    embedding_dim = 8


class RPSEmbedding(CustomEmbedding):
    allowed_values = ["Rock", "Paper", "Scissors"]
    storage_path = embeddings_path / "rps_embedding.pth"
    embedding_dim = 3


class YomiPlayerEmbedding(CustomEmbedding):
    allowed_values = [
        "Grave",
        "Jaina",
        "Argagarg",
        "Geiger",
        "BBB",
        "Setsuki",
        "Valerie",
        "Rook",
        "Midori",
        "Bigby",
        "Troq",
        "Onimaru",
        "Vendetta",
        "River",
        "Lum",
        "Degrey",
        "Menelker",
        "Persephone",
    ]
    storage_path = embeddings_path / "yomiplayer_embedding.pth"
    embedding_dim = 8


class GemEmbedding(CustomEmbedding):
    allowed_values = [
        "Red",
        "Green",
        "Blue",
        "Black",
        "White",
        "Purple",
        "Orange",
        "Diamond",
    ]
    storage_path = embeddings_path / "gem_embedding.pth"
    embedding_dim = 5


def create_card_index(cards1, cards2, decider):
    first_index = []
    second_index = []
    for i, card1 in enumerate(cards1):
        for j, card2 in enumerate(cards2):
            if decider(card1, card2):
                first_index.append(i)
                second_index.append(j)
    return torch.tensor([first_index, second_index], dtype=torch.long)


def create_card_index_int(cards1, cards2, decider, value):
    first_index = []
    second_index = []
    for i, card1 in enumerate(cards1):
        for j, card2 in enumerate(cards2):
            if decider(card1, card2) == value:
                first_index.append(i)
                second_index.append(j)
    return torch.tensor([first_index, second_index], dtype=torch.long)


def generate_sample_data():
    yomi_encoder = YomiEmbedding.load()
    player_encoder = YomiPlayerEmbedding.load()
    gem_encoder = GemEmbedding.load()

    my_hand = ["LB", "HB", "Thr", "Burst", "A", "B", "A"]
    my_discard = ["Super1", "Super2"]
    opponent_hand = ["LB", "HB", "Thr", "Burst", "unknown", "unknown", "unknown"]
    opponent_discard = ["Super1", "Super2"]
    my_player = ["Grave"]
    opponent_player = ["Jaina"]
    my_gem = ["Red"]
    opponent_gem = ["Blue"]

    # This should be replaced with a more complicated function
    def is_undecidable(cardlabel1, cardlabel2):
        return (cardlabel1 == "unknown") or (cardlabel2 == "unknown")

    def beat_decider(cardlabel1, cardlabel2):
        if is_undecidable(cardlabel1, cardlabel2):
            return False
        if cardlabel1 > cardlabel2:
            return True

        return False

    def loss_decider(cardlabel1, cardlabel2):
        if is_undecidable(cardlabel1, cardlabel2):
            return False
        if cardlabel1 < cardlabel2:
            return True
        return False

    yomi_data = HeteroData()
    yomi_data["my_hand"].x = yomi_encoder.encode(my_hand)
    yomi_data["my_discard"].x = yomi_encoder.encode(my_discard)
    yomi_data["opponent_hand"].x = yomi_encoder.encode(opponent_hand)
    yomi_data["opponent_discard"].x = yomi_encoder.encode(opponent_discard)
    yomi_data.my_health = torch.tensor(100)
    yomi_data.opponent_health = torch.tensor(100)
    yomi_data.my_player = player_encoder.encode(my_player)
    yomi_data.opponent_player = player_encoder.encode(opponent_player)
    yomi_data.my_gem = gem_encoder.encode(my_gem)
    yomi_data.opponent_gem = gem_encoder.encode(opponent_gem)
    yomi_data.target = 1

    yomi_data["my_hand", "beats", "opponent_hand"].edge_index = create_card_index(
        my_hand, opponent_hand, beat_decider
    )
    yomi_data["my_hand", "loses_to", "opponent_hand"].edge_index = create_card_index(
        my_hand, opponent_hand, loss_decider
    )

    return yomi_data


def generate_small_sample_data():
    yomi_encoder = YomiEmbedding.load()
    player_encoder = YomiPlayerEmbedding.load()
    gem_encoder = GemEmbedding.load()

    my_hand = ["LB", "HB", "Thr", "Burst"]
    my_discard = ["Super1", "Super2"]
    opponent_hand = ["LB", "HB", "Thr", "Burst", "unknown", "unknown", "unknown"]
    opponent_discard = ["Super1", "Super2"]
    my_player = ["Grave"]
    opponent_player = ["Jaina"]
    my_gem = ["Red"]
    opponent_gem = ["Blue"]

    # This should be replaced with a more complicated function
    def is_undecidable(cardlabel1, cardlabel2):
        return (cardlabel1 == "unknown") or (cardlabel2 == "unknown")

    def beat_decider(cardlabel1, cardlabel2):
        if is_undecidable(cardlabel1, cardlabel2):
            return False
        if cardlabel1 > cardlabel2:
            return True

        return False

    def loss_decider(cardlabel1, cardlabel2):
        if is_undecidable(cardlabel1, cardlabel2):
            return False
        if cardlabel1 < cardlabel2:
            return True
        return False

    yomi_data = HeteroData()
    yomi_data["my_hand"].x = yomi_encoder.encode(my_hand)
    yomi_data["my_discard"].x = yomi_encoder.encode(my_discard)
    yomi_data["opponent_hand"].x = yomi_encoder.encode(opponent_hand)
    yomi_data["opponent_discard"].x = yomi_encoder.encode(opponent_discard)
    yomi_data.my_health = torch.tensor(100)
    yomi_data.opponent_health = torch.tensor(100)
    yomi_data.my_player = player_encoder.encode(my_player)
    yomi_data.opponent_player = player_encoder.encode(opponent_player)
    yomi_data.my_gem = gem_encoder.encode(my_gem)
    yomi_data.opponent_gem = gem_encoder.encode(opponent_gem)
    yomi_data.target = 1

    yomi_data["my_hand", "beats", "opponent_hand"].edge_index = create_card_index(
        my_hand, opponent_hand, beat_decider
    )
    yomi_data["my_hand", "loses_to", "opponent_hand"].edge_index = create_card_index(
        my_hand, opponent_hand, loss_decider
    )

    return yomi_data


def rps_standard_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")]:
        return +1
    elif card_combo in [("Scissors", "Rock"), ("Rock", "Paper"), ("Paper", "Scissors")]:
        return -1
    else:
        return 0


def rps_non_standard_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Rock", "Scissors")]:
        return +1
    elif card_combo in [("Scissors", "Rock")]:
        return -1 / 2
    elif card_combo in [("Paper", "Rock"), ("Scissors", "Paper")]:
        return +1 / 2
    elif card_combo in [("Rock", "Paper"), ("Paper", "Scissors")]:
        return -1 / 2
    else:
        return 0


def rps_non_standard_payout_opponent(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Rock", "Scissors")]:
        return +1 / 2
    elif card_combo in [("Scissors", "Rock")]:
        return -1
    elif card_combo in [("Paper", "Rock"), ("Scissors", "Paper")]:
        return +1 / 2
    elif card_combo in [("Rock", "Paper"), ("Paper", "Scissors")]:
        return -1 / 2
    else:
        return 0


def get_payout_tensor(my_hand, opponent_action, payout_function):
    output_tensor = zeros((len(my_hand), 1))
    for i, my_card in enumerate(my_hand):
        output_tensor[i] = payout_function(my_card, opponent_action)
    return output_tensor


def generate_rps_sample(
    payout_function=rps_standard_payout, self_model=None, opponent_model=None
):
    rps_encoder = RPSEmbedding.load()
    my_hand = ["Rock", "Paper", "Scissors"]
    shuffle(my_hand)
    opponent_hand = ["Rock", "Paper", "Scissors"]
    shuffle(opponent_hand)

    rps_data = HeteroData()
    rps_data["my_hand"].x = rps_encoder.encode(my_hand)
    rps_data["opponent_hand"].x = rps_encoder.encode(opponent_hand)

    rps_data["my_hand", "beats", "opponent_hand"].edge_index = create_card_index_int(
        my_hand, opponent_hand, payout_function, 1
    )
    rps_data["my_hand", "loses_to", "opponent_hand"].edge_index = create_card_index_int(
        my_hand, opponent_hand, payout_function, -1
    )
    rps_data["my_hand"].choices = my_hand
    rps_data["opponent_hand"].choices = opponent_hand

    if opponent_model is None:
        opponent_action = choice(opponent_hand)
    else:
        options = [val[0] for val in opponent_model]
        probabilities = [val[1] for val in opponent_model]
        opponent_action = choices(options, probabilities)[0]

    if self_model is None:
        self_action = choice(my_hand)
    else:
        options = [val[0] for val in self_model]
        probabilities = [val[1] for val in self_model]
        self_action = choices(options, probabilities)[0]

    rps_data.self_action = self_action
    rps_data.opponent_action = opponent_action
    rps_data.my_utility = payout_function(self_action, opponent_action)
    #
    # payout = get_payout_tensor(my_hand, opponent_action, payout_function)
    # regret = payout - actual_payout
    #
    rps_data.payout = get_payout_tensor(my_hand, opponent_action, payout_function)
    # rps_data.regret = regret

    return rps_data


class CardDataset(Dataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
