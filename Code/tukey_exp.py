import os, sys, re, time, csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from word_lists import *
import two_unit_test
# from helper_tests import *


infile = '../Data/speech_w_data_example.csv'
vocab_csv = '../Data/list_of_1500words.csv'



# Calculates the line breaks for each month of data
# DO NOT USE FOR LARGE DATA: PANDAS RUNS OUT OF MEMORY
def calculate_line_breaks2(infile):
    df = pd.read_csv(infile, encoding = 'latin1')
    date_counts = [(str(name)[:6], group.shape[0]) for name,group in df.groupby('date')]
    dates = [date_counts[0][0]]
    breaks = [0]
    curr_count = 0
    curr_date = dates[0]
    for date,count in date_counts:
        if date == curr_date:
            curr_count += count
        else:
            dates.append(date)
            breaks.append(curr_count)
            curr_date = date
            curr_count += count
    return breaks, dates


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
    unit1 = unit1.loc[(unit1.party == parties[0]) & (unit1.chamber == chambers[0]), ['speech_id', 'speech']]
    unit2 = unit2.loc[(unit2.party == parties[1]) & (unit2.chamber == chambers[1]), ['speech_id', 'speech']]
    hc, features = two_unit_test(unit1, unit2, vocab_list)

    # Write results to file
    dates = [str(unit1.date)[:6], str(unit2.date)[:6]]
    with open('results_{}_{}.csv'.format(dates[0], dates[1])) as outfile:
        writer = csv.writer(outfile)
        line = [interval, dates[0], parties[0], chambers[0], dates[1], parties[1], chambers[1], hc]
        line.extend([feature for feature in HC.features])
        # line.extend([word for word in HC.words])
        writer.writerow(line)



# Calls sequence of experiments
def main():
    # Experimental parameters / setup
    a = time.time()
    line_breaks, dates = calculate_line_breaks(infile)
    intervals = [1, 6, 12]
    parties = ['D', 'R']
    chambers = ['H', 'H']
    congress_id = 114
    df = pd.read_csv(vocab_csv, encoding = 'latin1')
    vocab_list = list(df['word'])   # list of words to count
    ignore_list = words_to_ignore + function_words + additional_words1 + additional_words2 + singletons
    b = time.time()
    # print(line_breaks, dates)
    print("Time for setup is {0:.3f} seconds".format(b - a))


    # Calculate units
    numunits = len(dates)
    interval = 1
    for i in range(0, numunits, interval):
        for j in range(i + 1, numunits, interval):
            unit1 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[i], nrows = line_breaks[i+interval] - line_breaks[i], names = ['speech_id', 'date', 'congress_id', 'chamber', 'party', 'speech'])
            unit2 = 0
            if j >= numunits - interval:
                unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j], names = ['speech_id', 'date', 'congress_id', 'chamber', 'party', 'speech'])
            else:
                unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j], nrows = line_breaks[j+interval] - line_breaks[j], names = ['speech_id', 'date', 'congress_id', 'chamber', 'party', 'speech'])
            print("Comparing units in {} and {}...".format(dates[i], dates[j]))
            run_experiment(interval, unit1, unit2, parties, chambers, vocab_list, ignore_list)
            sys.exit(2)


if __name__ == '__main__':
    main()