
from scipy.stats import wilcoxon, shapiro, ttest_ind, ranksums, ttest_rel
import pandas as pd


class Stats_Warehouse():

    def __init__(self):
        None

    # check if the median of one sample 'a' is significantly different than another sample 'b'.
    # a and b are paired.
    # alternative may be either: 'greater', 'less' or 'two-sided'
    def calc_wilcoxon_signed_rank(self, a, b, alternative):
        # type: (pd.Series, pd.Series, str) -> tuple
        return wilcoxon(a, b, alternative=alternative)

    # it is assumed that a, b are not normally distributed.
    # alternative may be either: 'greater', 'less' or 'two-sided'
    def calc_wilcoxon_ranksum(self, a, b, alternative):
        return ranksums(a, b, alternative=alternative)

    # test if the data is normally distributed.
    def calc_shapiro_wilc(self,a):
        # type: (pd.Series) -> tuple
        return shapiro(a)

    # equal variance for both populations is assumed. alternative can be either of {‘two-sided’, ‘less’, ‘greater’}
    def calc_t_test(self, a, b, equal_var, alternative):
        return ttest_ind(a,b, equal_var=equal_var, alternative=alternative)

    def calc_paired_t_test(self, a, b, alternative):
        return ttest_rel(a, b, alternative=alternative)

    def calc_statistical_descriptives(self, df):
        # type: (pd.DataFrame) -> pd.DataFrame
        return df.describe()

