#Contains the function hc_vals, which computes hc statistic
import numpy as np
from collections import namedtuple

# evaluate HC stat and related quantities

# pv is is a list of p values
# alpha is fraction of p values to consider
# if interp == TRUE then interpolation of p-values around 0.5 is performed. Overrides truncation
# returns a named tuple 
def hc_vals(pv, alpha = 0.45, interp = False):
    pv = np.asarray(pv)
    pv = pv[~np.isnan(pv)]
    n = len(pv)
    uu = (np.arange(1,n+1) - 0.5) / np.float(n) #approximate expectation of p-values 
    ps = np.sort(pv) #sorted pvals
    ps_idx = np.argsort(pv)
    p_half = np.where(abs(ps - 0.5) < 0.05) #p-values that are too close to 0.5
    if interp == True and len(p.half) > 1:
        i1 = max(0,p_half[0]-1)
        i2 = min(p_half[-1]+1, len(ps)-1)
        sq = np.linspace(ps[i1],ps[i2], num = len(p_half)+2)
        ps[p_half] <- sq[1:(len(sq)-1)]
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
        
    #Define a namedtuple hc_tuple to store the results
    
    return hc_star, pv[i_max_star]
