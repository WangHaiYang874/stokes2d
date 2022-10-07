from pipe import *
from .let import Let
from utils import *


class AbstractPipe:

    lets: List[Let]

    def __init__(self, lets: List[Let]):
        self.lets = lets
