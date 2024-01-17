import numpy as np


class SupplyEgg:
    grab_x_position_mm: float
    grab_y_position_mm: float
    grab_j4_angle_degrees: float

    def __init__(self, x: float, y: float, j4: float) -> None:
        self.grab_x_position_mm = x
        self.grab_y_position_mm = y
        self.grab_j4_angle_degrees = j4


class SupplySystem:
    supply_eggs: list[SupplyEgg]
    next_egg_index: int

    SAFE_Z_POSITION_MM: float = -75
    APPROACH_Z_POSITION_MM: float = -125
    RELEASE_Z_POSITION_MM: float = -95
    GRAB_Z_POSITION_MM: float = -170
    GRAB_GRIPPER_VALUE: int = 135

    def __init__(self) -> None:
        self.supply_eggs = []
        self.next_egg_index = 0

        self.supply_eggs.append(SupplyEgg(x=230, y=60, j4=75))
        self.supply_eggs.append(SupplyEgg(x=280, y=50, j4=10))
        self.supply_eggs.append(SupplyEgg(x=230, y=10, j4=83))
        self.supply_eggs.append(SupplyEgg(x=280, y=0, j4=30))
        self.supply_eggs.append(SupplyEgg(x=220, y=-40, j4=91))
        self.supply_eggs.append(SupplyEgg(x=270, y=-50, j4=0))
