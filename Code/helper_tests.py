import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer

def two_unit_test(unit1, unit2, words_to_ignore):
    
    
    
    # Some toy values to get the rest working --- return an actual data frame of statistics
    d = {'uu': [0.1, 0.9, 0.3], 'zz': [-1.2, 0.2, 2], 'pp': [0.1, 0.4, 0.01], 'word': ['massive', 'comp', 'exp']}
    return pd.DataFrame(data = d)