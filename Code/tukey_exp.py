import os, sys, re, time, csv
import numpy as np 
import pandas as pd

# sys.path.append('./DEP')
import word_lists
from exp_methods import *


# Experimental parameters / setup
# skiplines = 55
# line_breaks, dates = calculate_line_breaks(infile, skiplines)
saved_breaks, saved_dates = word_lists.line_breaks, word_lists.dates
nummonths = len(saved_dates)
intervals = [1, 3, 6, 12]
df = pd.read_csv(vocab_csv, encoding = 'latin1')
vocab_list = list(df['word'])   # list of words to count
params = build_params(intervals, nummonths)


# Optimized for CJ parrun
for k in range(len(params)):
    param = params[k]
    run_experiment(infile, param[0], param[1], saved_breaks, saved_dates, vocab_list, param[2])
