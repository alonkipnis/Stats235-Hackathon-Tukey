
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import scipy.stats as stats
from HC_aux import hc_vals


stemmer = SnowballStemmer("english")


#Some helper functions for two_unit_test function

def get_p(freq_x,freq_y,total_x,total_y):
    p = (freq_x + freq_y)/np.float((total_x + total_y))
    return p

def get_se(p, total_x,total_y):
    se = np.sqrt(p * (1.0-p) * (1.0/total_x + 1.0/total_y))
    return se


def get_z_score(freq_x,freq_y,total_x,total_y,se):
    z_score = (freq_x /np.float(total_x) - freq_y/np.float(total_y))/np.float(se)
    return z_score

def get_pval(z_score):
    pval = 2 * stats.norm.cdf(-np.abs(z_score))
    return pval

def get_pval2(freq_x, freq_y,total_x,total_y):
    pval2 = stats.binom_test(x = freq_x, n = freq_x + freq_y, 
                                  p = (total_x - freq_x) / np.float((total_y + total_x - freq_x - freq_y)))
    return pval2


# Input: unit1, unit2, which are dataframes with columns: speech_id (integer) , speech (string)
# Input: context_words, non_context_words; these lists of words we will keep
# Output: a dataframe containing hc statistic, a boolean array features which is 1 if the feature was important,
#         and an array of corresponding words which were used for the HC evaluation

def two_unit_test(unit1,unit2, context_words,non_context_words):

    #Operations on unit1
    unit1 = unit1.drop('speech_id', axis = 1)
    unit1_words = pd.DataFrame(unit1.speech.str.split().tolist()).stack()
    unit1_words = unit1_words.reset_index()[[0]] #Keep the column 0, corresponding to 'word'
    unit1_words.columns = ['word']
    unit1_words['word'] = unit1_words['word'].str.extract(r'([a-zA-z]+)')
    unit1_words = unit1_words.dropna()
    unit1_words['word'] = unit1_words['word'].apply(stemmer.stem)
    unit1_words = unit1_words.apply(pd.value_counts) #after this, words become the index column, and the frequency becomes the word column
    unit1_words = unit1_words.reset_index()[['index', 'word']] 
    unit1_words['freq'] = unit1_words['word']
    unit1_words['word'] = unit1_words['index']
    unit1_words = unit1_words.drop('index', axis = 1)
    unit1_words['total'] = unit1_words['freq'].sum()

    #Same operations on unit2
    unit2 = unit2.drop('speech_id', axis = 1)
    unit2_words = pd.DataFrame(unit2.speech.str.split().tolist()).stack()
    unit2_words = unit2_words.reset_index()[[0]] #Keep the column 0, corresponding to 'word'
    unit2_words.columns = ['word']

    unit2_words['word'] = unit2_words['word'].str.extract(r'([a-zA-z]+)')
    unit2_words = unit2_words.dropna()
    unit2_words['word'] = unit2_words['word'].apply(stemmer.stem)
    unit2_words = unit2_words.apply(pd.value_counts) #after this, words become the index column, and the frequency becomes the word column
    unit2_words = unit2_words.reset_index()[['index', 'word']] 
    unit2_words['freq'] = unit2_words['word']
    unit2_words['word'] = unit2_words['index']
    unit2_words = unit2_words.drop('index', axis = 1)
    unit2_words['total'] = unit2_words['freq'].sum()
    
    #Joining unit1 and unit2 for the HC computation

    word_counts = pd.merge(unit1_words,unit2_words,on = 'word')
    word_counts = word_counts[word_counts['word'].isin(context_words + non_context_words)]
    word_counts['total'] = word_counts['total_x'] + word_counts['total_y']
    word_counts = word_counts[word_counts['freq_x'] + word_counts['freq_y'] > 1] #double-check this line

    word_counts['p'] = word_counts.apply(lambda row: get_p(row['freq_x'], 
                                                         row['freq_y'], 
                                                         row['total_x'],
                                                        row['total_y']), axis=1)
    word_counts['se'] = word_counts.apply(lambda row: get_se(row['p'], 
                                                         row['total_x'],
                                                        row['total_y']), axis=1)
    word_counts['z_score'] = word_counts.apply(lambda row: get_z_score(row['freq_x'],
                                                                 row['freq_y'],
                                                                 row['total_x'],
                                                                 row['total_y'],
                                                                 row['se']), axis = 1)
    word_counts['pval'] = word_counts.apply(lambda row: get_pval(row['z_score']),axis = 1)
    word_counts['pval2'] = word_counts.apply(lambda row: get_pval2(row['freq_x'], 
                                                         row['freq_y'], 
                                                         row['total_x'],
                                                        row['total_y']), axis=1)
    
    #Pass in pval2, from binomial test, into HC function
    hc_result = hc_vals(word_counts['pval2'], alpha = 0.4)
    features = np.where(word_counts['pval2'] <= hc_result.hc, 1, 0) 
    return pd.DataFrame({'hc': hc_result.hc, 'features': features, 'word': word_counts['word']})


