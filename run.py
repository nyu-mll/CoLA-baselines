from modules import Trainer

if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_datasets()
    trainer.load()
    trainer.train()
