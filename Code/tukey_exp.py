import os, sys, re, time, csv
import numpy as np 
import pandas as pd

# sys.path.append('./DEP')
import word_lists
from exp_methods import *

 
# Experimental parameters / setup
skiplines = 48     #  48 to start from 198701 --- beginning
                    # 195 to start from 200001 --- millenium
                    # 367 to start from 201501 --- 114th Congress
# line_breaks, dates = calculate_line_breaks(infile, skiplines)
saved_breaks, saved_dates = word_lists.line_breaks[skiplines:], word_lists.dates[skiplines:]
nummonths = len(saved_dates)
print("Starting from {}, with {} total months considered".format(saved_dates[0], nummonths))

df = pd.read_csv(vocab_csv, encoding = 'latin1')
vocab_list = list(df['word'])   # list of words to count

intervals = [6, 12]   # intervals used for text
params = build_params(intervals, nummonths)
print("Attempting to run {} jobs".format(len(params)))


# Optimized for CJ parrun
for k in range(len(params)):
    param = params[k]
    run_text_experiment(infile, param[0], param[1], saved_breaks, saved_dates, vocab_list, param[2])