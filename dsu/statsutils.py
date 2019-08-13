import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.stats import diagnostic
from scipy.stats import chi2

# Chi-square tests
def chi2_test_independence(series1, series2):
    print("Chi-square test of independence()")
    from scipy.stats import chi2_contingency
    result = chi2_contingency(series1, series2)
    print("Statistical degrees of freedom")
    print(result[2])
    print("Chi-square value")
    print(result[0])
    print("p-value")
    print(result[1])

def chisq_test(O, E, degree=3, sig_level=0.05):
    measured_val = sum( [(o - e)**2/e for (o, e) in zip(O, E)] )
    return chi2.cdf(measured_val, degree), chi2.sf(measured_val, degree)

#TODO: only the non-parametric ones used, check the rest andfigure out how to choose
# parameters(think kde estimator from sklearn)
CHECK_DISTS = ['norm', 'expon', 'logistic', 'cosine', 'cauchy',]
        # 'poisson', 'zipf', 'geom', 'hypergeom', 'randint', 'multivariate_normal', 'weibull_min', 'weibull_max',
        # 'logistic', 'chi', 'chi2', 'alpha', 'beta', 'bernoulli','binom', 'exponweib','exponpow'

def check_normality(series, name):
    print("Anderson-Darling normality test on %s "%name)
    print("Statistic: %f \n p-value: %f\n"%diagnostic.normal_ad(series))

def distribution_similarity(series, dist_type, test_type='ks'):
    from scipy import stats
    test_results = pd.DataFrame(columns=['distribution', 'statistic', 'p-value'])
    if not isinstance(dist_type, str):
        dist_type = dist_type.cdf
    if test_type=='ks':
        stat, pval = stats.kstest(series, dist_type)
        test_results.loc[0] = [dist_type, stat, pval]
    elif test_type =='wald':
        test_results.loc[0] = lm.wald_test(series, dist_type)
    else:
        raise "Unknown distribution similarity test type"
    return test_results

def check_distribution(series, test_type='ks'):
    test_results = pd.DataFrame(columns=['distribution', 'statistic', 'p-value'])
    for i, distribution in enumerate(CHECK_DISTS):
        res = distribution_similarity(series, distribution, test_type)
        test_results = test_results.append(res, ignore_index=True)
    test_results.sort_values(by=['p-value'])
    return test_results


def calculate_anova(df, targetCol, sourceCol):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    lm = ols('%s ~ C(%s, Sum) + c'% (targetCol, sourceCol),
            data=df).fit()
    table = anova_lm(lm, typ=2)
    return table

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

