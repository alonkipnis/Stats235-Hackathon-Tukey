import os, sys, re, time, csv
import numpy as np 
import pandas as pd

# sys.path.append('./DEP')
from exp_methods import *
from word_lists import *
from two_unit_test import two_unit_test


# Experimental parameters / setup
line_breaks, dates = calculate_line_breaks(infile, 55)
numunits = len(dates)
intervals = [1, 3, 6, 12]
df = pd.read_csv(vocab_csv, encoding = 'latin1')
vocab_list = list(df['word'])   # list of words to count
params = build_params(intervals, numunits)

# Optimized for CJ parrun
for i in range(len(params)):
    param = params[i]
    run_experiment(infile, param[0], param[1], line_breaks, dates, vocab_list, param[2])
