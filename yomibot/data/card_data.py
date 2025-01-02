import os
import torch
import numpy as np
import pandas as pd
from random import choice, choices, shuffle
from torch import zeros
from torch_geometric.data import HeteroData, Dataset
from torch.nn import Embedding
from yomibot.common.paths import embeddings_path
from yomibot.data.helpers import flatten_dict, unflatten_dict


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


class PennyEmbedding(CustomEmbedding):
    allowed_values = ["Odd", "Even"]
    storage_path = embeddings_path / "penny_embedding.pth"
    embedding_dim = 2


class StateEmbedding(CustomEmbedding):
    allowed_values = [1, 2, 3]
    storage_path = embeddings_path / "state_embedding.pth"
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


def penny_standard_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Odd", "Odd"), ("Even", "Even")]:
        return +1
    else:
        return -1


def penny_opponent_standard_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Odd", "Even"), ("Even", "Odd")]:
        return +1
    else:
        return -1


def penny_non_standard_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Even", "Even")]:
        return +4
    elif card_combo in [("Odd", "Odd")]:
        return +1
    else:
        return -1


def penny_opponent_standard_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Odd", "Even"), ("Even", "Odd")]:
        return +1
    else:
        return -1


def penny_even_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_1 == "Even":
        return 1
    else:
        return -1


def penny_odd_payout(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_1 == "Odd":
        return 1
    else:
        return -1


def penny_non_standard_payout_opponent(card_1, card_2):
    card_combo = (card_1, card_2)
    if card_combo in [("Even", "Even")]:
        return -4
    elif card_combo in [("Odd", "Odd")]:
        return -1
    else:
        return +1


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
    payout_function=rps_standard_payout,
    self_model=None,
    opponent_model=None,
    mirror=False,
    opp_payout_function=rps_standard_payout,
):
    rps_encoder = RPSEmbedding.load()
    my_hand = ["Rock", "Paper", "Scissors"]
    shuffle(my_hand)
    opponent_hand = ["Rock", "Paper", "Scissors"]
    shuffle(opponent_hand)

    rps_data = HeteroData()
    rps_data["my_hand"].x = rps_encoder.encode(my_hand)
    rps_data["opponent_hand"].x = rps_encoder.encode(opponent_hand)

    if mirror:
        other_data = HeteroData()
        other_data["my_hand"].x = rps_encoder.encode(opponent_hand)
        other_data["opponent_hand"].x = rps_encoder.encode(my_hand)

    rps_data["my_hand", "beats", "opponent_hand"].edge_index = create_card_index_int(
        my_hand, opponent_hand, payout_function, 1
    )
    rps_data["my_hand", "loses_to", "opponent_hand"].edge_index = create_card_index_int(
        my_hand, opponent_hand, payout_function, -1
    )
    rps_data["my_hand"].choices = my_hand
    rps_data["opponent_hand"].choices = opponent_hand

    if mirror:
        other_data[
            "my_hand", "beats", "opponent_hand"
        ].edge_index = create_card_index_int(
            opponent_hand, my_hand, opp_payout_function, 1
        )
        other_data[
            "my_hand", "loses_to", "opponent_hand"
        ].edge_index = create_card_index_int(
            opponent_hand, my_hand, opp_payout_function, -1
        )
        other_data["my_hand"].choices = opponent_hand
        other_data["opponent_hand"].choices = my_hand

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
    action_index = rps_data["my_hand"]["choices"].index(self_action)
    action_index_tensor = torch.zeros((1, 3))
    action_index_tensor[0, action_index] = 1
    rps_data.action_index = action_index_tensor
    rps_data.opponent_action = opponent_action
    rps_data.my_utility = payout_function(self_action, opponent_action)
    rps_data.payout = get_payout_tensor(my_hand, opponent_action, payout_function)

    if mirror:
        other_data.self_action = opponent_action
        action_index = other_data["my_hand"]["choices"].index(opponent_action)
        action_index_tensor = torch.zeros((1, 3))
        action_index_tensor[0, action_index] = 1
        other_data.action_index = action_index_tensor
        other_data.opponent_action = self_action
        other_data.my_utility = opp_payout_function(opponent_action, self_action)
        other_data.payout = get_payout_tensor(
            opponent_hand, self_action, opp_payout_function
        )
        return rps_data, other_data

    return rps_data


def get_expected_utility(action, opp_policy, payout_function):
    return sum(
        payout_function(action, opp_action) * opp_policy[opp_action]
        for opp_action in opp_policy
    )


def get_penny_regrets(choices, self_policy, opp_policy, payout_function):
    expected_utility = sum(
        self_policy[action] * get_expected_utility(action, opp_policy, payout_function)
        for action in choices
    )
    action_utilities = torch.tensor(
        [get_expected_utility(action, opp_policy, payout_function) for action in choices]
    ).reshape(-1, 1)
    regret = action_utilities - expected_utility
    return regret


class PennyData(HeteroData):
    penny_encoder = PennyEmbedding.load()
    state_encoder = StateEmbedding.load()

    @classmethod
    def from_values(
        cls,
        self_action,
        self_model,
        opp_action,
        opp_model,
        weight=1,
        state=None,
        payout_dictionary=None,
        actor_index=0,
        **kwargs,
    ):
        if state is None:
            state = choice([1, 2, 3])

        if payout_dictionary is None:
            payout_function = penny_standard_payout
        else:
            payout_function = payout_dictionary[state][actor_index]

        my_hand = ["Odd", "Even"]
        shuffle(my_hand)
        opponent_hand = ["Odd", "Even"]
        shuffle(opponent_hand)

        state_embedding = cls.state_encoder.encode([state])
        penny_data = cls()
        penny_data["my_hand"].x = cls.penny_encoder.encode(my_hand)
        penny_data["opponent_hand"].x = cls.penny_encoder.encode(opponent_hand)

        penny_data["my_hand"].choices = my_hand
        penny_data["opponent_hand"].choices = opponent_hand

        penny_data[
            "my_hand", "beats", "opponent_hand"
        ].edge_index = create_card_index_int(my_hand, opponent_hand, payout_function, 1)
        penny_data[
            "my_hand", "loses_to", "opponent_hand"
        ].edge_index = create_card_index_int(my_hand, opponent_hand, payout_function, -1)
        penny_data["my_hand"].policy = self_model
        penny_data["opponent_hand"].policy = opp_model

        penny_data.self_action = self_action
        action_index = penny_data["my_hand"]["choices"].index(self_action)
        action_index_tensor = torch.zeros((1, 2))
        action_index_tensor[0, action_index] = 1
        penny_data.action_index = action_index_tensor
        penny_data.opponent_action = opp_action
        penny_data.my_utility = payout_function(self_action, opp_action)
        penny_data.payout = get_payout_tensor(my_hand, opp_action, payout_function)
        penny_data.regret = get_penny_regrets(
            my_hand, self_model[state], opp_model[state], payout_function
        )
        penny_data.weight = weight
        penny_data.other_attributes = list(kwargs.keys())

        penny_data.state_label = state
        penny_data.state_x = state_embedding

        penny_data.actor_index = actor_index

        for key, value in kwargs.items():
            setattr(penny_data, key, value)

        return penny_data

    def serialise(self):
        my_policy = self["my_hand"].policy
        self_model = flatten_dict(my_policy, parent_key="self_policy")

        opp_policy = self["opponent_hand"].policy
        opp_model = flatten_dict(opp_policy, parent_key="other_policy")

        serial_dict = {
            "self_action": self.self_action,
            "opp_action": self.opponent_action,
            **self_model,
            **opp_model,
        }
        for key in self.other_attributes:
            serial_dict[key] = getattr(self, key)
        serial_dict["weight"] = self.weight
        serial_dict["state"] = self.state_label
        serial_dict["actor_index"] = self.actor_index
        return serial_dict

    @classmethod
    def deserialise(cls, container, payout_dictionary=None):
        self_model = unflatten_dict(
            {
                key: value
                for key, value in container.items()
                if key.startswith("self_policy")
            }
        )["self_policy"]
        self_model = {int(key): val for key, val in self_model.items()}

        opp_model = unflatten_dict(
            {
                key: value
                for key, value in container.items()
                if key.startswith("other_policy")
            }
        )["other_policy"]
        opp_model = {int(key): val for key, val in self_model.items()}

        self_action = container["self_action"]
        opp_action = container["opp_action"]
        state = container["state"]
        actor_index = container["actor_index"]
        kwargs = {
            key: value
            for key, value in container.items()
            if (
                (not key.startswith("self_policy"))
                and (not key.startswith("other_policy"))
                and (key not in ["self_action", "opp_action", "state", "actor_index"])
            )
        }
        return cls.from_values(
            self_action=self_action,
            opp_action=opp_action,
            self_model=self_model,
            opp_model=opp_model,
            payout_dictionary=payout_dictionary,
            state=state,
            actor_index=actor_index,
            **kwargs,
        )


def generate_penny_sample(
    self_model=None,
    opponent_model=None,
    mirror=False,
    state=None,
    payout_dictionary=None,
    **kwargs,
):
    if state is None:
        state = choice([1, 2, 3])

    if payout_dictionary is None:
        payout_function = penny_standard_payout
        opp_payout_function = penny_opponent_standard_payout
        payout_dictionary = {
            state: (penny_standard_payout, penny_opponent_standard_payout)
        }
    else:
        payout_function, opp_payout_function = payout_dictionary[state]

    if self_model is None:
        self_action = choice(["Odd", "Even"])
        self_model = {state: {"Odd": 0.5, "Even": 0.5} for state in [1, 2, 3]}
    else:
        options = [key for key in self_model[state].keys()]
        probabilities = [value for value in self_model[state].values()]
        self_action = choices(options, probabilities)[0]

    if opponent_model is None:
        opponent_action = choice(["Odd", "Even"])
        opponent_model = {state: {"Odd": 0.5, "Even": 0.5} for state in [1, 2, 3]}
    else:
        options = [key for key in opponent_model[state].keys()]
        probabilities = [value for value in opponent_model[state].values()]
        opponent_action = choices(options, probabilities)[0]
    penny_data = PennyData.from_values(
        self_action=self_action,
        self_model=self_model,
        opp_action=opponent_action,
        opp_model=opponent_model,
        state=state,
        payout_dictionary=payout_dictionary,
        actor_index=0,
    )

    if mirror:
        other_data = PennyData.from_values(
            self_action=opponent_action,
            self_model=opponent_model,
            opp_action=self_action,
            opp_model=self_model,
            payout_dictionary=payout_dictionary,
            actor_index=1,
            **kwargs,
        )
        return penny_data, other_data

    return penny_data


class CardDataset(Dataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
