import os, sys, re, time, csv
import numpy as np 
import pandas as pd

sys.path.append('./DEP')
import word_lists
from exp_methods import *

 
# Experimental parameters / setup
skiplines = 48      #  48 to start from 198701 --- beginning
                    # 195 to start from 200001 --- millenium
                    # 367 to start from 201501 --- 114th Congress
# line_breaks, dates = calculate_line_breaks(infile, skiplines)
saved_breaks, saved_dates = word_lists.line_breaks[skiplines:], word_lists.dates[skiplines:]
nummonths = len(saved_dates)
print("Starting from {}, with {} total months considered".format(saved_dates[0], nummonths))


topic_25_df = pd.read_csv(topic_25_csv, encoding = 'latin1')
topic_75_df = pd.read_csv(topic_75_csv, encoding = 'latin1')


intervals = [6, 12]         # intervals used for topics
params = build_params(intervals, nummonths, [topic_25_df, topic_75_df])
print("Attempting to run {} jobs".format(len(params)))


# Optimized for CJ parrun
for k in range(len(params)):
    param = params[k]
    run_topic_experiment(infile, param[0], param[1], saved_breaks, saved_dates, vocab_list, param[3], param[2])