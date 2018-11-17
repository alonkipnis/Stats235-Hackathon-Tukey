#!/usr/bin/env python
# coding: utf-8

# # Changes in appearances of words in speeches between parties and congresses

# In this example we detect changes between two corpus of speeches taken from congressional records. This example shows you how to:
# 
# - Load, arrange, and clean the data
# - Compute p-values
# - Use HC to detect changes

# In[6]:


# Load speech data
import pandas as pd
import two_unit_test
import nltk
nltk.download('punkt')
#raw_corpus = pd.read_csv("~/Stats285-Hack/Stats235-Hackathon-Tukey/Data/speech_w_data_example.csv", encoding = 'latin1')
raw_corpus = pd.read_csv("~/Data/speech_w_data.csv", encoding = 'latin1')
headers = list(raw_corpus)
# print(headers)


# In[2]:


# Select two units from raw corpus for comparison
unit1 = raw_corpus.loc[(raw_corpus.party == 'R') & (raw_corpus.chamber == 'H') & (raw_corpus.congress_id == 114), ['speech_id', 'speech']]
unit2 = raw_corpus.loc[(raw_corpus.party == 'D') & (raw_corpus.chamber == 'H') & (raw_corpus.congress_id == 114), ['speech_id', 'speech']]
# print(list(unit1))


# In[3]:


list_of_words = pd.read_csv("~/Data/list_of_1500words.csv", encoding = 'latin1', names = ['i','word']).iloc[:,1:2]


# In[10]:


hc, features = two_unit_test.two_unit_test(unit1,unit2, list_of_words=list_of_words['word'][1:])


# In[4]:


import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
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
        return [self.wnl.stem(t) for t in word_tokenize(doc)]


# Input: unit1, unit2, which are dataframes with columns: speech_id (integer) , speech (string)
# Input: context_words, non_context_words; these lists of words we will keep
# Output: a dataframe containing hc statistic, a boolean array features which is 1 if the feature was important,
#         and an array of corresponding words which were used for the HC evaluation

def two_unit_test(unit1,unit2, list_of_words):
    
    #Operations on unit1
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import pandas as pd

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
    features = hc_result.p_sorted_idx[:hc_result.i_max_star]
    return hc_result.hc, features


# In[11]:


print("HC score = {}".format(hc))

print("List of distinguishing words:")
# Which words cause the difference?
print(list_of_words.reindex(features))


# In[ ]:




