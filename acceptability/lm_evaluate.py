from acceptability.modules import LMEvaluator

if __name__ == '__main__':
    trainer = LMEvaluator()
    trainer.load()
    trainer.eval()
