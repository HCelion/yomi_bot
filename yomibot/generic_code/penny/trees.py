from copy import deepcopy
from functools import reduce
from random import choices
from yomibot.data.card_data import PennyData


class PennyGame:
    """A container class for the game nodes to reference commonly used items"""

    def __init__(self, model_dict, payout_dictionary, reservoir_dict):
        self.model_dict = model_dict
        self.payout_dictionary = payout_dictionary
        self.terminal_states = [2, 3]
        self.reservoir_dict = reservoir_dict


class PennyNode:
    def __init__(
        self,
        state,
        game,
        player_0_action=None,
        player_1_action=None,
        player=0,
        p0_probs=None,
        p1_probs=None,
        p0_payouts=None,
        p1_payouts=None,
        p0_update=None,
        p1_update=None,
    ):
        self.state = state
        self.available_actions = ["Even", "Odd"]
        self.player = player
        self.game = game  # Reference to models and payouts
        self.player0_action = player_0_action
        self.player1_action = player_1_action
        self.action_values = {"Even": None, "Odd": None}
        self.fill_policy()
        self.action_values = None
        self.has_snapshot = False
        if p0_payouts is None:
            p0_payouts = []
        if p1_payouts is None:
            p1_payouts = []
        if p0_probs is None:
            p0_probs = []
        if p1_probs is None:
            p1_probs = []

        self.p0_payouts = p0_payouts
        self.p1_payouts = p1_payouts
        self.p0_probs = p0_probs
        self.p1_probs = p1_probs

        if (player_0_action is not None) & (player_1_action is not None):
            p0_payout, p1_payout = self.get_immediate_payoff()
            self.p0_payouts += [p0_payout]
            self.p1_payouts += [-p0_payout]

        if p0_update is not None:
            self.p0_probs += [p0_update]

        if p1_update is not None:
            self.p1_probs += [p1_update]

        # if (not self.is_player_0()) and (not self.is_terminal()) and (not self.is_complete()):
        #     print(self, self.is_terminal())

    def __repr__(self):
        return (
            "PennyNode(State "
            + str(self.state)
            + ",Player "
            + str(self.player)
            + ","
            + str(self.player0_action)
            + ","
            + str(self.player1_action)
            + ",p0_payouts:"
            + str(self.p0_payouts)
            + ",p1_payouts:"
            + str(self.p1_payouts)
            + ",p1_probs:"
            + str(self.p0_probs)
            + ",p2_probs:"
            + str(self.p1_probs)
            + ")"
        )

    def fill_policy(self):
        player_model = self.game.model_dict[self.player]
        state_policy = player_model[self.state]
        self.policy = state_policy

    def is_complete(self):
        return (self.player0_action is not None) and (self.player1_action is not None)

    def get_immediate_payoff(self):
        player_0_payoff = self.game.payout_dictionary[self.state][0](
            self.player0_action, self.player1_action
        )
        player_1_payoff = -player_0_payoff
        return player_0_payoff, player_1_payoff

    def is_terminal(self):
        return self.is_complete() and (self.state in self.game.terminal_states)

    def is_player_0(self):
        return self.player == 0

    def snap_shot_state(self):
        if self.has_snapshot:
            return

        reservoir = self.game.reservoir_dict[self.player]
        self_model = self.policy
        opp_model = self.game.model_dict[1 - self.player][self.state]

        options = [key for key in self_model.keys()]
        probabilities = [value for value in self_model.values()]
        self_action = choices(options, probabilities)[0]

        options = [key for key in opp_model.keys()]
        probabilities = [value for value in opp_model.values()]
        opp_action = choices(options, probabilities)[0]

        state = PennyData.from_values(
            self_action=self_action,
            self_model=self.game.model_dict[self.player],
            opp_action=opp_action,
            opp_model=self.game.model_dict[1 - self.player],
            state=self.state,
            payout_dictionary=self.game.payout_dictionary,
            actor_index=self.player,
            regret_dict=self.get_regrets(),
        )
        reservoir.store_data([state])
        self.has_snapshot = True

    def sum_payouts(self):
        return sum(self.p0_payouts), sum(self.p1_payouts)

    def get_action_probability(self, action):
        return self.game.model_dict[self.player][self.state][action]

    def evaluate_actions(self):
        if self.action_values is not None:
            return self.action_values

        action_values = {}
        for action in self.available_actions:
            probability = self.get_action_probability(action)
            node = self.get_next_node(action, probability)

            if node.is_terminal():
                player_0_avg_value, player_1_avg_value = node.sum_payouts()
            else:
                (player_0_avg_value, player_1_avg_value) = node.get_average_value()
            if self.is_player_0():
                action_values[action] = player_0_avg_value
            else:
                action_values[action] = player_1_avg_value
        self.action_values = action_values
        if not (self.is_complete()):
            self.snap_shot_state()
        return action_values

    def get_average_value(self):
        if self.is_terminal():
            return self.sum_payouts()
        action_values = self.evaluate_actions()
        avg_value = 0
        for action in self.available_actions:
            avg_value += action_values[action] * self.policy[action]
        if self.is_player_0():
            return (avg_value, -avg_value)
        return (-avg_value, avg_value)

    def get_regret_prob_factor(self):
        if self.is_player_0():
            return reduce(lambda x, y: x * y, self.p1_probs, 1)
        return reduce(lambda x, y: x * y, self.p0_probs, 1)

    def get_regrets(self):
        prob_prefactor = self.get_regret_prob_factor()
        action_values = self.evaluate_actions()
        avg_value_p0, avg_value_p1 = self.get_average_value()
        if self.is_player_0():
            regret_dict = {
                key: prob_prefactor * (value - avg_value_p0)
                for key, value in action_values.items()
            }
        else:
            regret_dict = {
                key: prob_prefactor * (value - avg_value_p1)
                for key, value in action_values.items()
            }
        return regret_dict

    def get_next_node(self, action, probability):
        if not self.is_complete():
            if self.is_player_0():
                p0_update = probability
                p1_update = None
            else:
                p0_update = None
                p1_update = probability
        else:
            p0_update = None
            p1_update = None

        if self.is_terminal():
            return None
        if not self.is_complete():
            if self.player0_action is None:
                return PennyNode(
                    state=self.state,
                    game=self.game,
                    player_0_action=action,
                    player=1,
                    p0_payouts=deepcopy(self.p0_payouts),
                    p1_payouts=deepcopy(self.p1_payouts),
                    p0_update=p0_update,
                    p1_update=p1_update,
                    p0_probs=deepcopy(self.p0_probs),
                    p1_probs=deepcopy(self.p1_probs),
                )
            return PennyNode(
                state=self.state,
                game=self.game,
                player_0_action=self.player0_action,
                player_1_action=action,
                player=0,
                p0_payouts=deepcopy(self.p0_payouts),
                p1_payouts=deepcopy(self.p1_payouts),
                p0_update=p0_update,
                p1_update=p1_update,
                p0_probs=deepcopy(self.p0_probs),
                p1_probs=deepcopy(self.p1_probs),
            )

        # If not is not terminal or complete, we must be in state 1
        # and the game transitions to a new node
        if self.player0_action == "Even":
            return PennyNode(
                state=2,
                game=self.game,
                player_0_action=None,
                player=0,
                p0_payouts=deepcopy(self.p0_payouts),
                p1_payouts=deepcopy(self.p1_payouts),
                p0_update=p0_update,
                p1_update=p1_update,
                p0_probs=deepcopy(self.p0_probs),
                p1_probs=deepcopy(self.p1_probs),
            )
        return PennyNode(
            state=3,
            game=self.game,
            player_0_action=None,
            player=0,
            p0_payouts=deepcopy(self.p0_payouts),
            p1_payouts=deepcopy(self.p1_payouts),
            p0_update=p0_update,
            p1_update=p1_update,
            p0_probs=deepcopy(self.p0_probs),
            p1_probs=deepcopy(self.p1_probs),
        )
