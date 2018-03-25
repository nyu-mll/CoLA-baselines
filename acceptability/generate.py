from acceptability.modules import LMGenerator

if __name__ == '__main__':
    trainer = LMGenerator()
    trainer.load()
    trainer.generate()
