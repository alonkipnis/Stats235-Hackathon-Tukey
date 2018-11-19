
import pandas as pd
import numpy as np
from HC_aux import hc_vals

from sklearn.feature_extraction.text import CountVectorizer

import re
import snowballstemmer 
class LemmaTokenizer(object):
    def __init__(self):
         self.wnl = snowballstemmer.stemmer('english')
    def __call__(self, doc):
        doc = re.sub(r'[^A-Za-z0-9\s]',r' ',doc)
        doc = re.sub(r'\n',r' ',doc)
        doc = re.sub(r'[0-9]',r' ',doc)
        #doc = re.sub(r'[a-z]\040' ,r'',doc) #remove singletons
        return self.wnl.stemWords(doc.split())

import scipy.stats as stats
from HC_aux import hc_vals



#Some helper functions for two_unit_test function

def get_z_score(freq_x,freq_y,total_x,total_y):
    p = (freq_x + freq_y)/np.float((total_x + total_y))
    se = np.sqrt(p * (1.0-p) * (1.0/total_x + 1.0/total_y))
    z_score = (freq_x /np.float(total_x) - freq_y/np.float(total_y))/np.float(se)
    return z_score

def get_pval(z_score):
    pval = 2 * stats.norm.cdf(-np.abs(z_score))
    return pval

def get_pval2(freq_x, freq_y,total_x,total_y, min_counts = 10):
    pval2 = np.nan
    if (freq_x + freq_y >= min_counts):
        pval2 = stats.binom_test(x = freq_x, n = freq_x + freq_y, 
                                  p = (total_x - freq_x) / np.float((total_y + total_x - freq_x - freq_y)))
    return pval2



def two_unit_test(unit1,unit2, list_of_words):
# Input: unit1, unit2, which are dataframes with columns: speech_id (integer) , speech (string)
# Input: lists_of_words = features
# Output: hc score, list of distinguishing features

    tf_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), vocabulary=list_of_words)
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
                                                        row['T2']), axis=1)
    
    #Pass in pval2, from binomial test, into HC function
    hc_star, p_val_star  = hc_vals(word_counts['pval'], alpha = 0.25)
    word_counts['flag'] = word_counts['pval'] < p_val_star
    features = np.where(word_counts['flag'] == True)
    return (hc_star, features)


