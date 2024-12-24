from dataclasses import dataclass


@dataclass
class Individual:
    id: int
    position: tuple[int, int]
