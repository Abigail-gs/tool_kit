
from evaluation_warehouse import Evaluation_API_Warehouse
from statistics_warehuose import Stats_Warehouse
import torch



class Evaluation_API():

    def __init__(self):
        None

    # predictions are the logits of the ModelOutput; output = (loss, logits)
    def get_ck(self, predictions, targets, weights):
        # type: (tuple, torch.tensor, str) -> float
        return Evaluation_API_Warehouse.calc_cohen_kappa(predictions, targets, weights)

    def get_confmat(self, predictions, labels):
        # type: (tuple, torch.tensor) -> float
        return Evaluation_API_Warehouse.calc_confmat(predictions, labels)

    def get_f1(self, predictions, labels):
        # type: (tuple, torch.tensor) -> float
        return Evaluation_API_Warehouse.calc_f1_score(predictions, labels)

    def get_descriptives(self, df):
        # type: (pd.DataFrame) -> pd.DataFrame
        return Stats_Warehouse.calc_statistical_descriptives(df)

    def get_shapiro_wilc(self,predictions, labels):
        # type: (tuple, torch.tensor) -> float
        return Stats_Warehouse.calc_shapiro_wilc(predictions, labels)

    def get_t_test(self, a, b, alternative):
        # type: (list, list, str) -> tuple
        return Stats_Warehouse.calc_t_test(a, b, alternative)

    def get_paired_t_test(self, a, b, alternative):
        # type: (list, list, str) -> tuple
        return Stats_Warehouse.calc_paired_t_test(self, a, b, alternative)

