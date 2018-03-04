import torch
import os


class Checkpoint:
    def __init__(self, args):
        self.args = args
        self.experiment_ckpt_path = os.path.join(self.args.save_loc,
                                                 self.args.experiment_name + ".ckpt")

    def load_state_dict(self, model):
        # TODO: Add logging for checkpoint loaded in both

        # First check if resume arg has been passed
        # TODO: Add resume to parse_args
        if self.args.resume and os.path.exists(self.args.resume):
            model.load_state_dict(torch.load(self.args.resume))

        # Then check if current experiement has a checkpoint
        elif os.path.exists(self.experiment_ckpt_path):
            model.load_state_dict(torch.load(self.experiment_ckpt_path))

    def save(self, model):
        if not os.path.exists(os.path.dirname(self.experiment_ckpt_path)):
            os.mkdir(os.path.dirname(self.experiment_ckpt_path))
        torch.save(model.state_dict(), self.experiment_ckpt_path)

    def restore(self, model):
        if os.path.exists(self.experiment_ckpt_path):
            model.load_state_dict(torch.load(self.experiment_ckpt_path))
