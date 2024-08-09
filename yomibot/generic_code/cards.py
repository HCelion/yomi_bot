from __future__ import annotations


class Card:
    def __init__(
        self,
        speed: Integer,
        damage: Integer,
        card_type: String,
    ) -> None:
        """
        Class for containing card attributes.
        Primary use is opener resolution.
        Secondary use is exchanges, power ups, etc.

        ...

        Attributes:
        ----------
        card_type:
            One of A-G, LB, HB, Dodge, Thr, X-Z, Abi1/2,  Burst, Special1/2.
        speed:
            Necessary but not sufficient component of opener resolution.
            Between card matches for throws and attacks, the value with the highest amount wins.
        damage:
            If applicable, the int amount to substract from opponent health.

        Methods
        -------
        __eq__(other):
            Compares one card to another for power ups.

        TODO:
        ----------
            Add the below to Card definition when we are ready
            combo_type: String,
            combo_points: Integer,
            cost: Integer = 0,
            block_damage: = 0, # May change this default later

            self.combo_type = combo_type
            self.combo_points = combo_points
            self.cost = cost
            self.block_damage = block_damage
        """
        self.type = card_type
        self.damage = damage
        self.speed = speed

    def __eq__(self, other):
        if isinstance(other, Card):
            assert self.type == other.type
            assert self.damage == other.damage
            assert self.speed == other.speed
            return True
        return False
