
import pandas as pd
import numpy as np
from HC_aux import hc_vals

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import re
from nltk import word_tokenize          
from nltk.stem import SnowballStemmer 
import Stemmer
class LemmaTokenizer(object):
    def __init__(self):
         self.wnl = Stemmer.Stemmer('english')
    def __call__(self, doc):
        doc = re.sub(r'[^A-Za-z0-9\s]',r' ',doc)
        doc = re.sub(r'\n',r' ',doc)
        doc = re.sub(r'[0-9]',r' ',doc)
        #doc = re.sub(r'[a-z]\040' ,r'',doc) #remove singletons
        return word_tokenize(self.wnl.stemWord(doc))

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

def get_pval2(freq_x, freq_y,total_x,total_y):
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
    word_counts = word_counts[word_counts['n1'] + word_counts['n2'] >= 10] 
    word_counts['pval'] = word_counts.apply(lambda row: get_pval2(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2']), axis=1)
    
    #Pass in pval2, from binomial test, into HC function
    hc_result = hc_vals(word_counts['pval'], alpha = 0.4)
    features_idx = hc_result.p_sorted_idx[:hc_result.i_max_star]
    features = [list(word_counts['word'])[idx] for idx in features_idx]
    features_idx_original = [idx for (idx,val) in enumerate(list_of_words) if val in features]
    return hc_result.hc, features_idx_original


