import pandas as pd
import nltk
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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

import re
from nltk import word_tokenize          
from nltk.stem import SnowballStemmer 
class LemmaTokenizer(object):
    def __init__(self):
         self.wnl = SnowballStemmer(language = 'english')
    def __call__(self, doc):
        doc = re.sub(r'[^A-Za-z0-9\s]',r' ',doc)
        doc = re.sub(r'\n',r' ',doc)
        doc = re.sub(r'[0-9]',r' ',doc)
        #doc = re.sub(r'[a-z]\040' ,r'',doc) #remove singletons
        return word_tokenize(self.wnl.stem(doc))

def two_unit_test_full(unit1,unit2, list_of_words):
# Input: unit1, unit2, which are dataframes with columns: speech_id (integer) , speech (string)
# Input: lists_of_words = features
# Output: full list of counts + pvals + hc
    import scipy.stats as stats
    def get_pval2(n1, n2, T1 ,T2, min_counts = 1):
        if (n1 + n2 >= min_counts) :
            pval = stats.binom_test(x = n1, n = n1 + n2, 
                                      p = (T1 - n1) / np.float((T1 + T1 - n1 - n2)))
        else :
            pval = np.nan
        return pval

    tf_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),vocabulary=list_of_words)
    tf1 = tf_vectorizer.fit_transform(unit1['speech'])
    tf2 = tf_vectorizer.fit_transform(unit2['speech'])

    word_counts = pd.DataFrame()
    word_counts['word'] = (tf_vectorizer.get_feature_names())
    word_counts['n1'] = np.array(tf1.sum(0)).T
    word_counts['n2'] = np.array(tf2.sum(0)).T
    word_counts['T1'] = word_counts['n1'].sum()
    word_counts['T2'] = word_counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    word_counts['pval'] = word_counts.apply(lambda row: get_pval2(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'], min_counts = 10), axis=1)

    #Pass in pval2, from binomial test, into HC function
    hc_star, p_val_star = hc_vals(word_counts['pval'], alpha = 0.4)
    word_counts['hc'] = hc_star
    word_counts['flag'] = word_counts['pval'] < p_val_star
    return word_counts
