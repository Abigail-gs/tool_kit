
import torch
from torchmetrics import CohenKappa, ConfusionMatrix


class Evaluation_API_Warehouse():

    def __init__(self):
        None

    def calc_cohen_kappa(self, predictions, targets, weights):
        if weights == None:
            pred_flat = torch.argmax(predictions, dim=1)
            w_cohenkappa = CohenKappa(task="binary", num_classes=2)  # (num_classes=2)
        elif weights == 'quadratic':
            pred_flat = torch.argmax(predictions, dim=1)
            w_cohenkappa = CohenKappa(num_classes=2, weights='quadratic')
        return float(w_cohenkappa(pred_flat, targets))

    def calc_confmat(self, predictions, labels):
        confmat = ConfusionMatrix(task="binary", num_classes=2)
        return confmat(torch.argmax(predictions, axis=1), labels)

    def calc_f1_score(self, predictions, labels):
        confmat = self.calc_confmat(predictions, labels)
        fp = float(confmat[0][1])
        fn = float(confmat[1][0])
        tp = float(confmat[1][1])

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        return f1_score, precision, recall
