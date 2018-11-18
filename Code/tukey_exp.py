import os, sys, re, time, csv
import numpy as np 
import pandas as pd

sys.path.append('./DEP')
import word_lists
from exp_methods import *


# Experimental parameters / setup
skiplines = 48
# line_breaks, dates = calculate_line_breaks(infile, skiplines)
saved_breaks, saved_dates = word_lists.line_breaks[skiplines:], word_lists.dates[skiplines:]
nummonths = len(saved_dates)
df = pd.read_csv(vocab_csv, encoding = 'latin1')
vocab_list = list(df['word'])   # list of words to count
topic_25_list = pd.read_csv(topic_25_csv, encoding = 'latin1')
topic_75_list = pd.read_csv(topic_75_csv, encoding = 'latin1')


# intervals = [1, 3, 6, 12]   # intervals used for text
# params = build_params(intervals, nummonths)
intervals = [3, 12]         # intervals used for topics
params = build_params(intervals, nummonths, [topic_25_list, topic_75_list])


# Optimized for CJ parrun
for k in range(len(params)):
    param = params[k]
    # run_text_experiment(infile, param[0], param[1], saved_breaks, saved_dates, vocab_list, param[2])
    run_topic_experiment(infile, param[0], param[1], saved_breaks, saved_dates, param[3], param[2])