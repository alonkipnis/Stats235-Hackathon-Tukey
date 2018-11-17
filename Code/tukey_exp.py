import os, sys, re, time, csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from word_lists import *
from two_unit_test import two_unit_test
# from helper_tests import *


infile = '../Data/speech_w_data_example.csv'
interval = 1
outfile = '../Data/results_{}month.csv'.format(interval)
vocab_csv = '../Data/list_of_1500words.csv'
fieldnames = ["Interval", "Date1", "Party1", "Affil1", "Date2", "Party2", "Affil2", "HC_score", "Features"]


# Calculates the line breaks for each month of data
def calculate_line_breaks(infile):
    dates = []
    line_breaks = []
    date = '0'
    linenum = 0
    with open(infile) as csvfile:
        data = csv.reader(csvfile)
        next(data)
        for row in data:
            curr_date = str(row[1])[:6]
            if curr_date == date:
                linenum += 1
            else:
                dates.append(curr_date)
                line_breaks.append(linenum)
                date = curr_date
                linenum += 1
    return line_breaks, dates



# Run an experiment with a set of parameters
def run_experiment(interval, unit1, unit2, parties, chambers, vocab_list, ignore_list):
    comp_unit1 = unit1.loc[(unit1.party == parties[0]) & (unit1.chamber == chambers[0]), ['speech_id', 'speech']]
    comp_unit2 = unit2.loc[(unit2.party == parties[1]) & (unit2.chamber == chambers[1]), ['speech_id', 'speech']]
    hc, features = two_unit_test(comp_unit1, comp_unit2, vocab_list)

    # Write results to file
    dates = [str(unit1.date[2])[:6], str(unit2.date[2])[:6]]
    with open(outfile, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        line = [interval, dates[0], parties[0], chambers[0], dates[1], parties[1], chambers[1], hc, ",".join([str(f) for f in features])]
        writer.writerow(line)
        print("Results written to CSV")



# Calls sequence of experiments
def main():
    # Experimental parameters / setup
    a = time.time()
    line_breaks, dates = calculate_line_breaks(infile)
    intervals = [1, 6, 12]
    parties = ["D", "R"]
    chambers = ["H", "H"]
    congress_id = 114
    df = pd.read_csv(vocab_csv, encoding = 'latin1')
    vocab_list = list(df['word'])   # list of words to count
    ignore_list = words_to_ignore + function_words + additional_words1 + additional_words2 + singletons
    b = time.time()
    # print(line_breaks, dates)
    with open(outfile, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
    print("Time for setup is {0:.3f} seconds".format(b - a))


    # Calculate units
    numunits = len(dates)
    for i in range(0, numunits, interval):
        for j in range(i + interval, numunits, interval):
            a = time.time()
            unit1 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[i], nrows = line_breaks[i+interval] - line_breaks[i], names = ['speech_id', 'date', 'congress_id', 'chamber', 'party', 'speech'])
            unit2 = 0
            if j >= numunits - interval:
                unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j], names = ['speech_id', 'date', 'congress_id', 'chamber', 'party', 'speech'])
            else:
                unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j], nrows = line_breaks[j+interval] - line_breaks[j], names = ['speech_id', 'date', 'congress_id', 'chamber', 'party', 'speech'])
            print("Comparing units in {} and {}...".format(dates[i], dates[j]))
            run_experiment(j - i, unit1, unit2, parties, chambers, vocab_list, ignore_list)
            b = time.time()
            print("Time for running 1 iteration is {0:.3f} seconds".format(b - a))



if __name__ == '__main__':
    main()