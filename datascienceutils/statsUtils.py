import statsmodels.api as sm
import numpy as np
from scipy.stats import chi2
#print(chi2.cdf(ch, num_sides-1), chi2.sf(ch, num_sides-1))

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

