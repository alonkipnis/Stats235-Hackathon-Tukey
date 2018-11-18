import os, sys, re, time, csv
import numpy as np 
import pandas as pd
from exp_methods import *
from word_lists import *
from two_unit_test import two_unit_test
skiplines = 55
line_breaks, dates = calculate_line_breaks(infile, skiplines)
numunits = len(dates)
intervals = [1, 3, 6, 12]
df = pd.read_csv(vocab_csv, encoding = 'latin1')
vocab_list = list(df['word'])   # list of words to count
params = build_params(intervals, numunits)
    
i_fid = open('/tmp/i.tmp','w')
for i in range(len(params)):i_fid.write("%i\n" % i);
i_fid.close()
