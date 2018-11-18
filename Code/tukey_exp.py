import os, sys, re, time, csv
import numpy as np 
import pandas as pd

from exp_methods import *
from word_lists import *
from two_unit_test import two_unit_test


# Experimental parameters / setup
line_breaks, dates = calculate_line_breaks(infile)
numunits = len(dates)
intervals = [1, 3, 6, 12]
df = pd.read_csv(vocab_csv, encoding = 'latin1')
vocab_list = list(df['word'])   # list of words to count
params = build_params(intervals, numunits)

# Optimized for CJ parrun
for param in params:
    run_experiment(infile, param[0], param[2]-param[1], param[1], param[2], line_breaks, dates, vocab_list, param[3])
