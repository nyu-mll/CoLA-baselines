import math
import numpy as np
from torchnet import meter


class Meter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = meter.ConfusionMeter(self.num_classes)

    def confusion(self):
        return self.confusion_matrix.conf

    def add(self, pred, target):
        self.confusion_matrix.add(pred, target)

    def _get_fps(self):
        conf = self.confusion()
        tp = int(conf[0][0])
        fp = int(conf[0][1])
        fn = int(conf[1][0])
        tn = int(conf[1][1])
        return tp, fp, fn, tn

    def matthews(self):
        tp, fp, fn, tn = self._get_fps()
        try:
            m = float((tp * tn) - (fp * fn)) / \
                math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        except ZeroDivisionError:
            m = 0
        return m

    def f1(self):
        tp, fp, fn, tn = self._get_fps()
        try:
            f = float(2 * tp) / float((2 * tp) + fp + fn)
        except ZeroDivisionError:
            f = 0
        return f

    def accuracy(self):
        tp, fp, fn, tn = self._get_fps()
        return float(tp + tn) / float(tp + fp + tn + fn)

    def reset(self):
        self.confusion_matrix = meter.ConfusionMeter(self.num_classes)
