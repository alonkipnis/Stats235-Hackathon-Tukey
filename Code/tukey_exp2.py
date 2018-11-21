import os, sys, re, time, csv
import numpy as np 
import pandas as pd

on_cluster = True
on_cluster = False
resnames = ["Interval", "Date1", "Party1", "Affil1", "Date2", "Party2", "Affil2", "HC_score", "Features"]
datanames = ['speech_id', 'date', 'congress_id', 'chamber', 'party',
             'tf-idf', 'topic25', 'topic50', 'topic75', 'topic75_top3']
infile, vocab_csv, topic_25_csv, topic_75_csv = None, None, None, None
if on_cluster:
    infile = os.path.expanduser('~/Data/speeches_1464words_tfidf.csv')
    vocab_csv = os.path.expanduser('~/Data/1464words_unstemmed.csv')
    #topic_25_csv = os.path.expanduser('~/Data/word_25topics_LDA.csv')
    #topic_75_csv = os.path.expanduser('~/Data/word_75topics_LDA.csv')
else:
    sys.path.append('./DEP')
    #infile = '../Data/speech_w_data.csv'
    infile = '../Data/speeches_1464words_tfidf.csv'
    #vocab_csv = '../Data/alt_list_of_words.csv'
    vocab_csv = '../Data/1464words_unstemmed.csv'
    #topic_25_csv = '../Data/word_25topics_LDA.csv'
    #topic_75_csv = '../Data/word_75topics_LDA.csv'

import word_lists
from exp_methods import *
 
# Experimental parameters / setup
skiplines =  48     #  48 to start from 198701 --- beginning
                    # 195 to start from 200001 --- millenium
                    # 367 to start from 201501 --- 114th Congress
# line_breaks, dates = calculate_line_breaks(infile, skiplines)
saved_breaks, saved_dates = word_lists.line_breaks[skiplines:], word_lists.dates[skiplines:]
nummonths = len(saved_dates)
print("Starting from {}, with {} total months considered".format(saved_dates[0], nummonths))

df = pd.read_csv(vocab_csv, encoding = 'latin1')
vocab_list = list(df['term'])   # list of words to count

intervals = [6, 12]   # intervals used for text
params = build_params(intervals, nummonths)
print("Attempting to run {} jobs".format(len(params)))

# Optimized for CJ parrun
for k in range(len(params)):
    param = params[k]
    #run_text_experiment(infile, param[0], param[1], saved_breaks, saved_dates, vocab_list, param[2], datanames)
    run_experiment2(infile, param[0], param[1], saved_breaks, saved_dates, vocab_list, param[2], datanames)
