import statsmodels.api as sm
import numpy as np
from scipy.stats import chi2
#print(chi2.cdf(ch, num_sides-1), chi2.sf(ch, num_sides-1))

def chisq_stat(O, E):
    return sum( [(o - e)**2/e for (o, e) in zip(O, E)] )

def chisq_test(O, E, degree=3, sig_level=0.05):
    measured_val = sum( [(o - e)**2/e for (o, e) in zip(O, E)] )
    return chi2.cdf(measured_val, degree), chi2.sf(measured_val, degree)


# Simulate R's poly function
# from [here.](http://stackoverflow.com/questions/41317127/python-equivalent-to-r-poly-function)

def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]

