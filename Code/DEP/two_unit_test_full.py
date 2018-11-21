import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

def hc_vals(pv, alpha = 0.45):
    pv = np.asarray(pv)
    pv = pv[~np.isnan(pv)]
    n = len(pv)
    uu = (np.arange(1,n+1) - 0.5) / np.float(n) #approximate expectation of p-values 
    ps = np.sort(pv) #sorted pvals
    ps_idx = np.argsort(pv)
    
    z_alt = (uu - ps) / np.sqrt(ps * (1 - ps) + 0.001 ) * np.sqrt(n); #zeroth order HC approach (can be extended) 
    z = (uu - ps)/np.sqrt(uu * (1 - uu)) * np.sqrt(n)

    i_lim = int(np.floor(alpha * n + 0.5))
    i_max = np.argmax(z[:i_lim])
    z_max = z[i_max]

    if i_max + 1 == 1: #if optimal is at the first entry
        i_max_star = 1 + np.argmax(z[1:i_lim])
        hc_star    = z[i_max_star]
    else:
        i_max_star = i_max
        hc_star    = z_max

    p_star = ps[i_max_star]
    
    i_max = np.argmax(z_alt[:i_lim])
    z_max_alt = z_alt[i_max]
    if i_max + 1 == 1: #if optimal is at the first entry
        i_max_star = 1 + np.argmax(z_alt[1:i_lim])
        hc_alt    = z_alt[i_max_star]
    else:
        i_max_star = i_max
        hc_alt    = z_max_alt    

    return hc_star, p_star, hc_alt


def get_pval_bin(n1, n2, T1 ,T2, min_counts = 2):
        from scipy.stats import binom_test
        if (n1 + n2 >= min_counts) :
            pval = binom_test(x = n1, n = n1 + n2, 
                                      p = (T1 - n1) / np.float((T1 + T1 - n1 - n2)))
        else :
            pval = np.nan
        return pval

def get_pval_z(n1, n2, T1 ,T2, min_counts = 2):
    from scipy.stats import norm
    def get_z_score(freq_x,freq_y,total_x,total_y):
        p = (freq_x + freq_y)/np.float((total_x + total_y))
        se = np.sqrt(p * (1.0-p) * (1.0/total_x + 1.0/total_y))
        z_score = (freq_x /np.float(total_x) - freq_y/np.float(total_y))/np.float(se)
        return z_score
    
    if (n1 + n2 >= min_counts) :
        z_score = get_z_score(n1, n2, T1, T2)
        pval = 2 * norm.cdf(-np.abs(z_score))
    else :
        pval = np.nan
    return pval

import re     
import snowballstemmer
class StemTokenizer(object):
    def __init__(self):
         self.wnl = snowballstemmer.stemmer('english')
    def __call__(self, doc):
        doc = re.sub(r'[^A-Za-z0-9\s]',r' ',doc)
        doc = re.sub(r'\n',r' ',doc)
        doc = re.sub(r'[0-9]',r' ',doc)
        #doc = re.sub(r'[a-z]\040' ,r'',doc) #remove singletons
        return self.wnl.stemWords(doc.split())

class myTokenizer(object):
    def __call__(self, doc):
        doc = re.sub(r'[^A-Za-z0-9\s]',r' ',doc)
        #doc = re.sub(r'[a-z]\040' ,r'',doc) #remove singletons
        return doc.split()


def two_unit_test_words_full(unit1, unit2, vocabulary, alpha = 0.45, min_counts = 5):
# Input: unit1, unit2, which are dataframes with columns: speech_id (integer) , speech (string)
# Input: vocabulary = features
# Output: data frame: "word, n1, T1, n2, T2, pval, pval_z, hc"

    tf_vectorizer = CountVectorizer(tokenizer=StemTokenizer(),vocabulary=vocabulary)
    tf1 = tf_vectorizer.fit_transform(unit1['speech'])
    tf2 = tf_vectorizer.fit_transform(unit2['speech'])

    counts = pd.DataFrame()
    counts['word'] = (tf_vectorizer.get_feature_names())
    counts['n1'] = np.array(tf1.sum(0)).T
    counts['n2'] = np.array(tf2.sum(0)).T
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    counts['pval'] = counts.apply(lambda row: get_pval_bin(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'], 
                                                        min_counts = min_counts), axis=1)
    counts['pval_z'] = counts.apply(lambda row: get_pval_z(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'], 
                                                        min_counts = min_counts), axis=1)

    hc_star, p_val_star, hc_star_alt = hc_vals(counts['pval'], alpha = alpha)
    counts['hc'] = hc_star
    counts['hc_alt'] = hc_star_alt
    counts['flag'] = counts['pval'] < p_val_star
    return counts

def test_tfidf(unit1, unit2, vocab, min_counts = 25, ignore_list = [], alpha = 0.45) :
    # Input: unit1, unit2, are dataframes with column: 'tf-idf'
    # Input: vocab = list of words 
    # Output: data frame: "word, n1, T1, n2, T2, pval, pval_z, hc"

    from sklearn.feature_extraction.text import CountVectorizer
    tf_vectorizer = CountVectorizer(vocabulary=vocab)
    
    tf1 = tf_vectorizer.fit_transform(unit1['tf-idf'])
    tf2 = tf_vectorizer.fit_transform(unit2['tf-idf'])

    counts = pd.DataFrame()
    counts['term'] = vocab
    counts['n1'] = np.array(tf1.sum(0))[0]
    counts['n2'] = np.array(tf2.sum(0))[0]
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    counts['pval'] = counts.apply(lambda row: get_pval_bin(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'],
                                                    min_counts = min_counts), axis=1)

    counts['pval_z'] = counts.apply(lambda row: get_pval_z(row['n1'], 
                                                             row['n2'], 
                                                             row['T1'],
                                                            row['T2'], 
                                                        min_counts = min_counts), axis=1)
    
    pv = counts[~counts['term'].isin(ignore_list)]['pval']
    hc_star, p_val_star, hc_star_alt = hc_vals(pv, alpha = alpha)
    counts['hc'] = hc_star
    counts['hc_alt'] = hc_star_alt
    counts['flag'] = counts['pval'] < p_val_star 
    counts.loc[counts['term'].isin(ignore_list),'flag'] = np.nan
    return counts


def test_topics(unit1, unit2, by, min_counts = 25, ignore_list = [], alpha = 0.45) :
    # Input: unit1, unit2, are dataframes 
    # by is name of column to count
    # Output: data frame: "topic, n1, T1, n2, T2, pval, pval_z, hc, hc_alt"

    cnt1 = unit1[by].value_counts()
    df1 = pd.DataFrame({'topic' : cnt1.index, 'count' : cnt1.values})

    cnt2 = unit2[by].value_counts()
    df2 = pd.DataFrame({'topic' : cnt2.index, 'count' : cnt2.values})

    cnt = pd.DataFrame()
    cnt['topic'] = range(75)
    counts = pd.concat([cnt, cnt1, cnt2], axis = 1, join = 'inner', names = ['topic', 'n1', 'n2'])
    counts.columns = ['topic', 'n1', 'n2']
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    counts['pval'] = counts.apply(lambda row: get_pval_bin(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'],
                                                        min_counts = min_counts), axis=1)

    counts['pval_z'] = counts.apply(lambda row: get_pval_z(row['n1'], 
                                                             row['n2'], 
                                                             row['T1'],
                                                            row['T2'], 
                                                            min_counts = min_counts), axis=1)
    
    pv = counts[~counts['topic'].isin(ignore_list)]['pval']
    hc_star, p_val_star, hc_star_alt = hc_vals(pv, alpha = alpha)
    counts['hc'] = hc_star
    counts['hc_alt'] = hc_star_alt
    counts['flag'] = counts['pval'] < p_val_star 
    counts.loc[counts['topic'].isin(ignore_list),'flag'] = np.nan
    return counts

def two_unit_test_topics_full(unit1,unit2, term_topic_df, ignore_topics = [], alpha = 0.45, min_counts = 5) :
    # Input: unit1, unit2, which are dataframes with columns: speech_id (integer) , speech (string)
    # Input: vocabulary = features
    # Output: data frame: "word, n1, T1, n2, T2, pval, pval_z, hc"

    tm_tp_mat = term_topic_df.iloc[:,2:].values
    vocab = term_topic_df['term']
    T = np.shape(tm_tp_mat)[1] #num of topics
    tf_vectorizer = CountVectorizer(tokenizer=StemTokenizer(), vocabulary=vocab) 
    tf1 = tf_vectorizer.fit_transform(unit1['speech'])
    tf2 = tf_vectorizer.fit_transform(unit2['speech'])

    tf1_speech_topic = (tf1.dot(tm_tp_mat)).argmax(axis = 1) #value in each row is topic doc is assigned to
    tf2_speech_topic = (tf2.dot(tm_tp_mat)).argmax(axis = 1)

    tf1_topics, n1 = np.unique(tf1_speech_topic, return_counts = True)
    tf2_topics, n2 = np.unique(tf2_speech_topic, return_counts = True)


    counts = pd.DataFrame()
    counts['topic'] = np.arange(T)
    counts['n1']= np.zeros(T)
    counts['n2'] = np.zeros(T)
    counts.loc[tf1_topics,'n1'] = n1
    counts.loc[tf2_topics,'n2'] = n2
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    counts['pval'] = counts.apply(lambda row: get_pval_bin(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'],
                                                        min_counts = min_counts), axis=1)

    counts['pval_z'] = counts.apply(lambda row: get_pval_z(row['n1'], 
                                                             row['n2'], 
                                                             row['T1'],
                                                            row['T2'], 
                                                            min_counts = min_counts), axis=1)

    #Pass in pval from binomial test into HC function
    pv = counts[~counts['topic'].isin(ignore_topics)]['pval']
    hc_star, p_val_star, hc_star_alt = hc_vals(pv, alpha = alpha)
    counts['hc'] = hc_star
    counts['hc_alt'] = hc_star_alt
    counts['flag'] = counts['pval'] < p_val_star 
    counts.loc[counts['topic'].isin(ignore_topics),'flag'] = np.nan
    return counts

def test_words(unit1, unit2, vocab, min_counts = 25, ignore_list = [], alpha = 0.45) :
    # Input: unit1, unit2, are dataframes with column: 'tf-idf'
    # Input: vocab = list of words 
    # Output: data frame: "word, n1, T1, n2, T2, pval, pval_z, hc"

    from sklearn.feature_extraction.text import CountVectorizer
    tf_vectorizer = CountVectorizer(vocabulary=vocab)
    
    tf1 = tf_vectorizer.fit_transform(unit1['tf-idf'])
    tf2 = tf_vectorizer.fit_transform(unit2['tf-idf'])

    counts = pd.DataFrame()
    counts['term'] = vocab
    counts['n1'] = np.array(tf1.sum(0))[0]
    counts['n2'] = np.array(tf2.sum(0))[0]
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    counts['pval'] = counts.apply(lambda row: get_pval_bin(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'],
                                                        min_counts = min_counts), axis=1)

    counts['pval_z'] = counts.apply(lambda row: get_pval_z(row['n1'], 
                                                             row['n2'], 
                                                             row['T1'],
                                                            row['T2'], 
                                                            min_counts = min_counts), axis=1)
    
    pv = counts[~counts['term'].isin(ignore_list)]['pval']
    hc_star, p_val_star, hc_star_alt = hc_vals(pv, alpha = alpha)
    counts['hc'] = hc_star
    counts['hc_alt'] = hc_star_alt
    counts['flag'] = counts['pval'] < p_val_star 
    counts.loc[counts['term'].isin(ignore_list),'flag'] = np.nan
    return counts

def test_topics_top3(unit1, unit2, min_counts = 25, ignore_list = [], alpha = 0.45) :
    from sklearn.feature_extraction.text import CountVectorizer
    tf_vectorizer = CountVectorizer(tokenizer=myTokenizer(),  vocabulary=[str(i) for i in range(75)])
    tf1 = tf_vectorizer.fit_transform(unit1['topic75_top3'])
    tf2 = tf_vectorizer.fit_transform(unit2['topic75_top3'])
    
    counts = pd.DataFrame()
    counts['topic'] = range(75)
    counts['n1'] = np.array(tf1.sum(0))[0]
    counts['n2'] = np.array(tf2.sum(0))[0]
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    counts['pval'] = counts.apply(lambda row: get_pval_bin(row['n1'], 
                                                         row['n2'], 
                                                         row['T1'],
                                                        row['T2'],
                                                  min_counts = min_counts), axis=1)

    counts['pval_z'] = counts.apply(lambda row: get_pval_z(row['n1'], 
                                                             row['n2'], 
                                                             row['T1'],
                                                            row['T2'], 
                                                            min_counts = min_counts), axis=1)
    
    pv = counts[~counts['topic'].isin(ignore_list)]['pval']
    hc_star, p_val_star, hc_star_alt = hc_vals(pv, alpha = alpha)
    counts['hc'] = hc_star
    counts['hc_alt'] = hc_star_alt
    counts['flag'] = counts['pval'] < p_val_star 
    counts.loc[counts['topic'].isin(ignore_list),'flag'] = np.nan
    return counts


def get_topic(unit, term_topic_df) :
    #infer topic of unit from term_topic dataframe
    tm_tp_mat = term_topic_df.iloc[:,2:].values
    vocab = term_topic_df['term']
    T = np.shape(tm_tp_mat)[1] #num of topics
    tf_vectorizer = CountVectorizer(tokenizer=StemTokenizer(), vocabulary=vocab) 
    tf = tf_vectorizer.fit_transform(unit['speech'])

    tf_speech_topic = (tf.dot(tm_tp_mat)).argmax(axis = 1) #value in each row is topic doc is assigned to
    
    return tf_speech_topic


