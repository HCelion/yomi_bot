from torch import optim
import numpy as np


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):  # pylint: disable=(protected-access)
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def get_nash_equilibria(A, return_value=False):
    import nashpy as nash

    rps = nash.Game(A, -A)
    eqs = rps.support_enumeration()
    player_1, player_2 = list(eqs)[0]
    order = ("Rock", "Paper", "Scissors")
    optimum_1 = {action: prob for action, prob in zip(order, player_1)}
    optimum_2 = {action: prob for action, prob in zip(order, player_2)}

    if return_value:
        p1_value = np.array(player_1).T @ A @ np.array(player_2)
        return optimum_1, optimum_2, p1_value
    return optimum_1, optimum_2
