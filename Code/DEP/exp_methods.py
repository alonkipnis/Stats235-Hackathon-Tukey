import os, sys, re, time, csv
import numpy as np 
import pandas as pd

from word_lists import *
from two_unit_test import two_unit_test


#infile = '../Data/speech_w_data_example.csv'
#vocab_csv = '../Data/list_of_1500words.csv'
infile = os.path.expanduser('~/Data/speech_w_data.csv')
vocab_csv = os.path.expanduser('~/Data/list_of_1500words.csv')
resnames = ["Interval", "Date1", "Party1", "Affil1", "Date2", "Party2", "Affil2", "HC_score", "Features"]
datanames = ['speech_id', 'date', 'congress_id', 'chamber', 'party', 'speech']


# Calculates the line breaks for each month of data
def calculate_line_breaks(infile):
    dates = []
    line_breaks = []
    date = '0'
    linenum = 0
    with open(infile, 'r') as csvfile:
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



# Make parameters to loop through
def build_params(intervals, numunits):
    params_list = []
    for interval in intervals:
        for i in range(0, numunits - 2 * interval):
            for j in range(i + interval, numunits - interval):
                params_list.append([interval, i, j, ['N', 'N']])
                params_list.append([interval, i, j, ['D', 'D']])
                params_list.append([interval, i, j, ['D', 'R']])
                params_list.append([interval, i, j, ['R', 'D']])
                params_list.append([interval, i, j, ['R', 'R']])
    return params_list



# Run an experiment with a set of parameters
def run_experiment(infile, interval, diff, i, j, line_breaks, dates, vocab_list, parties):
    print("Comparing units in {} and {} between {}...".format(dates[i], dates[j], parties))
    unit1 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[i], nrows = line_breaks[i+interval] - line_breaks[i], names = datanames)
    unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j], nrows = line_breaks[j+interval] - line_breaks[j], names = datanames)

    # Stop early if there's not enough speeches
    if unit1.shape[0] < 400 or unit2.shape[0] < 400:
        return
    
    comp_unit1, comp_unit2 = None, None
    if parties == ['N', 'N']:
        comp_unit1 = unit1.loc[:, ['speech_id', 'speech']]
        comp_unit2 = unit2.loc[:, ['speech_id', 'speech']]
    else:
        comp_unit1 = unit1.loc[(unit1.party == parties[0]), ['speech_id', 'speech']]
        comp_unit2 = unit2.loc[(unit2.party == parties[1]), ['speech_id', 'speech']]
    hc, features = two_unit_test(comp_unit1, comp_unit2, vocab_list)

    # Write results to file
    #outfile = '../Data/results_{}_{}_{}.csv'.format(interval, parties[0], parties[1])
    outfile = os.path.expanduser('~/Data/results_{}_{}_{}.csv'.format(interval, parties[0], parties[1]))
    dates = [str(unit1.date[2])[:6], str(unit2.date[2])[:6]]
    with open(outfile, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        line = [diff, dates[0], parties[0], dates[1], parties[1], hc, ','.join([str(f) for f in features])]
        writer.writerow(line)
