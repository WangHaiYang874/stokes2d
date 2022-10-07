from .let_index import LetIndex
from dataclasses import dataclass


@dataclass(order=True, repr=True, frozen=True, init=False)
class Vertex:

    l1: LetIndex
    l2: LetIndex

    def __init__(self, l1, l2):
        assert l1.pipeIndex != l2.pipeIndex

        l1, l2 = sorted([l1, l2])
        object.__setattr__(self, 'l1', l1)
        object.__setattr__(self, 'l2', l2)

    @property
    def atBdr(self):
        return self.l1.atBdr or self.l2.atBdr
