from typing import Protocol


class HasPosition(Protocol):
    position: tuple[int, int]