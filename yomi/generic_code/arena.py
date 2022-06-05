from abc import ABC, abstractmethod


class Arena(ABC):
    @property
    @abstractmethod
    def player_generator(self):
        pass

    def __init__(
        self,
        left_strategy="best_play",
        right_strategy="right_play",
        left_generator=None,
        right_generator=None,
    ):
        if left_generator:
            self.left_player = left_generator(side="left", strategy=left_strategy)
        else:
            self.left_player = self.player_generator(
                side="left", strategy=left_strategy
            )

        if right_generator:
            self.right_player = right_generator(side="right", strategy=right_strategy)
        else:
            self.right_player = self.player_generator(
                side="right", strategy=right_strategy
            )

    def play_game(self):

        while not self.is_game_end(self.left_player, self.right_player):
            # TODO if choosing the card gets expensive, we have to run this
            # async so that they can run in parallel
            left_card = self.left_player.choose_card()
            right_card = self.right_player.choose_card()
            outcome = self.evaluate_round_outcome(left_card, right_card)
            self.left_player.update_state(outcome)
            self.right_player.update_state(outcome)

        # TODO Add players logging their experiences
        game_outcome = self.evaluate_game_outcome(self.left_player, self.right_player)
        return game_outcome

    @abstractmethod
    def is_game_end(self, left_player, right_player):
        pass

    @abstractmethod
    def evaluate_round_outcome(left_card, right_card):
        pass

    @abstractmethod
    def evaluate_game_outcome(self, left_player, right_player):
        pass
