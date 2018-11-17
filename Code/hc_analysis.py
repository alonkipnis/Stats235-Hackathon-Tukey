import os, sys, re, time, csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from word_lists import *
from two_unit_test import two_unit_test


outfile = '../Data/results_1month.csv'
vocab_csv = '../Data/list_of_1500words.csv'




# Analyze the results by plotting HC over time
def hc_plot(df):
    fig, ax = plt.subplots()
    fig2 = plt.plot(df.Interval, df.HC_score, 'bo')
    plt.xlabel('Days Apart')
    plt.ylabel('HC score')
    plt.savefig('../Data/HC_vs_time_interval.png')



def main():
    vocab_df = pd.read_csv(vocab_csv, encoding = 'latin1')
    vocab_list = list(vocab_df['word']) 
    results_df = pd.read_csv(outfile)
    print(results_df.head(1))
    # hc_plot(results_df)



if __name__ == '__main__':
    main()