import torch
import os

from ..utils.flags import get_parser
from ..utils.general import get_model_instance
from ..utils.checkpoint import Checkpoint


class Generator:
    def __init__(self):
        parser = get_parser()
        self.args = parser.parse_args()
