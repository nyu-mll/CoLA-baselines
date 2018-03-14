# pylint: disable=E0401
from modules import LMTrainer

if __name__ == '__main__':
    trainer = LMTrainer()
    trainer.load()
    trainer.train()
