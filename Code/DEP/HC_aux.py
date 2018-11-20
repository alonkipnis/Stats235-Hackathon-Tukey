#Contains the function hc_vals, which computes hc statistic
import numpy as np

# evaluate HC stat and related quantities

# pv is is a list of p values
# alpha is fraction of p values to consider
# if interp == TRUE then interpolation of p-values around 0.5 is performed. Overrides truncation
# returns a named tuple 
def hc_vals(pv, alpha = 0.45):
    pv = np.asarray(pv)
    pv = pv[~np.isnan(pv)]
    n = len(pv)
    uu = (np.arange(1,n+1) - 0.5) / np.float(n) #approximate expectation of p-values 
    ps = np.sort(pv) #sorted pvals
    ps_idx = np.argsort(pv)
    
    #z = (uu - ps) / np.sqrt(ps * (1 - ps) + 0.01 ) * sqrt(n); #zeroth order HC approach (can be extended) 
    z = (uu - ps)/np.sqrt(uu * (1 - uu)) * np.sqrt(n)

    max_i = int(np.floor(alpha * n + 0.5))
    i_max = np.argmax(z[:max_i])
    z_max = z[i_max]

    if i_max + 1 == 1: #if optimal is at the first entry
        i_max_star = 1 + np.argmax(z[1:max_i])
        hc_star    = z[i_max_star]
    else:
        i_max_star = i_max
        hc_star    = z_max
        
    return hc_star, ps[i_max_star]