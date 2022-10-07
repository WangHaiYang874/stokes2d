from dataclasses import dataclass


@dataclass(order=True, repr=True, frozen=True)
class LetIndex:

    pipeIndex: int
    letIndex:  int

    def __getitem__(self, i: int):
        return [self.pipeIndex, self.letIndex][i]

    @property
    def atBdr(self):
        return self.pipeIndex == -1
