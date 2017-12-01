import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.stats import diagnostic
from scipy.stats import chi2

#TODO: only the non-parametric ones used, check the rest andfigure out how to choose
# parameters(think kde estimator from sklearn)
CHECK_DISTS = ['norm', 'expon', 'logistic', 'cosine', 'cauchy',]
        # 'poisson', 'zipf', 'geom', 'hypergeom', 'randint', 'multivariate_normal', 'weibull_min', 'weibull_max',
        # 'logistic', 'chi', 'chi2', 'alpha', 'beta', 'bernoulli','binom', 'exponweib','exponpow'
def check_normality(series, name):
    print("Anderson-Darling normality test on %s "%name)
    print("Statistic: %f \n p-value: %f\n"%diagnostic.normal_ad(series))

def is_independent(series1, series2):
    pass

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

def is_similar_distribution(original_dist, target_dist, test_type='permutation'):
    if test_type=='permutation':
        from permute.core import two_sample
        kwargs = {'stat':'t','alternative':'two-sided','seed':20}
        p_value = two_sample(original_dist, target_dist)
        print(p_value)
    elif test_type=='chi_sq':
        pass
    else:
        raise "Unknown distribution similarity test type"

def distribution_tests(series, test_type='ks', dist_type=None):
    from scipy import stats
    if not dist_type:
        test_results = pd.DataFrame(columns=['distribution', 'statistic', 'p-value'])
        for i, distribution in enumerate(CHECK_DISTS):
            if test_type=='ks':
                print("Kolmogrov - Smirnov test with distribution %s"%distribution)
                stat, pval = stats.kstest(series, distribution)
                test_results.loc[i] = [distribution, stat, pval]
            elif test_type =='wald':
                print("Wald test with distribution %s"%distribution)
                print(lm.wald_test(series, distribution))
            else:
                raise "Unknown distribution similarity test type"
    else:
        test_results.loc[0] = [dist_type, stats.kstest(series, dist_type)]
    return test_results

def chisq_stat(O, E):
    return sum( [(o - e)**2/e for (o, e) in zip(O, E)] )

def chisq_test(O, E, degree=3, sig_level=0.05):
    measured_val = sum( [(o - e)**2/e for (o, e) in zip(O, E)] )
    return chi2.cdf(measured_val, degree), chi2.sf(measured_val, degree)

def calculate_anova(df, targetCol, sourceCol):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    lm = ols('%s ~ C(%s, Sum) + c'% (targetCol, sourceCol),
            data=df).fit()
    table = anova_lm(lm, typ=2)
    return table

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

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

# Simulate R's poly function
# from [here.](http://stackoverflow.com/questions/41317127/python-equivalent-to-r-poly-function)

def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]

