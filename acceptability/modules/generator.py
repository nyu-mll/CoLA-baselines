import torch
import os

from acceptability.utils.flags import get_parser
from acceptability.utils.general import get_model_instance
from acceptability.utils.checkpoint import Checkpoint


class Generator:
    def __init__(self):
        parser = get_parser()
        self.args = parser.parse_args()
