import os, sys, re, time, csv
import numpy as np 
import pandas as pd

from two_unit_test import two_unit_test, two_unit_test_topics
from two_unit_test_full import test_tfidf, test_topics, test_topics_top3

# Calculates the line breaks for each month of data
def calculate_line_breaks(infile, skip_lines):
    dates = []
    line_breaks = []
    date = '0'
    linenum = 0
    with open(infile, 'rt', encoding = 'latin1') as csvfile:
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
    if skip_lines > len(dates):
        skip_lines = 0
    return line_breaks[skip_lines:], dates[skip_lines:]



# Make parameters to loop through for text test
def build_params(intervals, nummonths):
    params_list = []
    for interval in intervals:
        for i in range(0, nummonths - 2 * interval, interval):
        # for i in range(0, nummonths - interval, interval):
            params_list.append([interval, i, ['N', 'N']])
            params_list.append([interval, i, ['D', 'D']])
            params_list.append([interval, i, ['D', 'R']])
            params_list.append([interval, i, ['R', 'D']])
            params_list.append([interval, i, ['R', 'R']])
    return params_list


# Make parameters to loop through for topic test
def build_topic_params(intervals, nummonths, topic_lists):
    params_list = []
    for interval in intervals:
        for i in range(0, nummonths - 2 * interval, interval):
            for topic in topic_lists:
                params_list.append([interval, i, ['N', 'N'], topic])
                params_list.append([interval, i, ['D', 'D'], topic])
                params_list.append([interval, i, ['D', 'R'], topic])
                params_list.append([interval, i, ['R', 'D'], topic])
                params_list.append([interval, i, ['R', 'R'], topic])
    return params_list


# Run an experiment with a set of parameters
def run_text_experiment(infile, interval, i, line_breaks, dates, vocab_list, parties, datanames):
    for j in range(i + interval, len(dates) - interval, interval):
    # for j in range(i, len(dates) - interval, interval):
        print("Comparing text in units {} and {} between {} parties...".format(dates[i], dates[j], parties))
        unit1 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[i], nrows = line_breaks[i+interval] - line_breaks[i], names = datanames)
        unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j], nrows = line_breaks[j+interval] - line_breaks[j], names = datanames)

        # Stop early if there's not enough speeches
        speech_count_thresh = 300
        if unit1.shape[0] < speech_count_thresh or unit2.shape[0] < speech_count_thresh:
            continue
        
        comp_unit1, comp_unit2 = None, None
        if parties == ['N', 'N']:
            comp_unit1 = unit1.loc[:, ['speech_id', 'speech']]
            comp_unit2 = unit2.loc[:, ['speech_id', 'speech']]
        else:
            comp_unit1 = unit1.loc[(unit1.party == parties[0]), ['speech_id', 'speech']]
            comp_unit2 = unit2.loc[(unit2.party == parties[1]), ['speech_id', 'speech']]

        # print(comp_unit1.shape, comp_unit2.shape, len(vocab_list))
        hc, features = None, None
        try:
            hc, features = two_unit_test(comp_unit1, comp_unit2, vocab_list)
        except:
            continue

        # Write results to file
        outfile = None
        if on_cluster:
            outfile = os.path.expanduser('~/Data/results_{}_{}_{}.csv'.format(interval, parties[0], parties[1]))
        else:
            outfile = '../Data/results_{}_{}_{}.csv'.format(interval, parties[0], parties[1])
        unit_dates = [str(unit1.date[2])[:6], str(unit2.date[2])[:6]]
        with open(outfile, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            line = [j - i, unit_dates[0], parties[0], unit_dates[1], parties[1], hc, ','.join([str(f) for f in features])]
            writer.writerow(line)

def run_experiment2(infile, interval, i, line_breaks, dates,
                     vocab, parties, datanames, on_cluster = False):
    for j in range(i + interval, len(dates) - interval, interval):
    # for j in range(i, len(dates) - interval, interval):
        print("Comparing text in units {} and {} between {} parties...".format(dates[i], dates[j], parties))
        unit1 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[i],
             nrows = line_breaks[i+interval] - line_breaks[i], names = datanames)
        unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j],
             nrows = line_breaks[j+interval] - line_breaks[j], names = datanames)

        # Stop early if there's not enough speeches
        speech_count_thresh = 300
        if unit1.shape[0] < speech_count_thresh or unit2.shape[0] < speech_count_thresh:
            continue
        
        comp_unit1, comp_unit2 = None, None
        if parties == ['N', 'N']:
            comp_unit1 = unit1.loc[:, ['speech_id', 'tf-idf', 'topic25', 'topic50', 'topic75' ,'topic75_top3']]
            comp_unit2 = unit2.loc[:, ['speech_id', 'tf-idf', 'topic25', 'topic50', 'topic75' ,'topic75_top3']]
        else:
            comp_unit1 = unit1.loc[(unit1.party == parties[0]), ['speech_id', 'tf-idf', 'topic25', 'topic50', 'topic75' ,'topic75_top3']]
            comp_unit2 = unit2.loc[(unit2.party == parties[1]), ['speech_id', 'tf-idf', 'topic25', 'topic50', 'topic75' ,'topic75_top3']]

        # tf-idf word counting test
        hc_words, hc_words_alt = None, None
        try:
            hc_words, hc_words_alt = list(test_tfidf(unit1, unit2, vocab = vocab).loc[0,['hc', 'hc_alt']])
        except:
            continue

        # topic counting test
        hc_t25, hc_t50, hc_t75 = None, None, None
        hc_t25_alt, hc_t50_alt, hc_t75_alt = None, None, None
        hc_t75_top3, hc_t75_top3_alt = None, None
        try:
            hc_t25, hc_t25_alt = list(test_topics(unit1, unit2, by = 'topic25').loc[0,['hc', 'hc_alt']])
            hc_t50, hc_t50_alt = list(test_topics(unit1, unit2, by = 'topic50').loc[0,['hc', 'hc_alt']])
            hc_t75, hc_t75_alt = list(test_topics(unit1, unit2, by = 'topic75').loc[0,['hc', 'hc_alt']])
        except:
            continue

        # top_3 topic counting test
        try:
            hc_t75_top3, hc_t75_top3_alt = list(test_topics_top3(unit1, unit2).loc[0,['hc', 'hc_alt']])
        except:
            continue


        # Write results to file
        outfile = None
        if on_cluster:
            outfile = os.path.expanduser('~/Data/results_{}_{}_{}.csv'.format(interval, parties[0], parties[1]))
        else:
            outfile = '../Data/results_{}_{}_{}.csv'.format(interval, parties[0], parties[1])
        unit_dates = [str(unit1.date[2])[:6], str(unit2.date[2])[:6]]
        with open(outfile, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            line = [unit_dates[0], parties[0], unit_dates[1], parties[1], 
                    hc_words, hc_words_alt, hc_t25, hc_t25_alt, hc_t50,
                     hc_t50_alt, hc_t75, hc_t75_alt, hc_t75_top3, hc_t75_top3_alt]
            writer.writerow(line)


# Run an experiment with NN clustering for topic assignment
def run_topic_experiment(infile, interval, i, line_breaks, dates, topics_df, parties, datanames, on_cluster):
    for j in range(i + interval, len(dates) - interval, interval):
        print("Comparing topic in units {} and {} between {} parties...".format(dates[i], dates[j], parties))
        unit1 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[i], nrows = line_breaks[i+interval] - line_breaks[i], names = datanames)
        unit2 = pd.read_csv(infile, encoding = 'latin1', skiprows = line_breaks[j], nrows = line_breaks[j+interval] - line_breaks[j], names = datanames)

        # Stop early if there's not enough speeches
        speech_count_thresh = 300
        if unit1.shape[0] < speech_count_thresh or unit2.shape[0] < speech_count_thresh:
            continue

        comp_unit1, comp_unit2 = None, None
        if parties == ['N', 'N']:
            comp_unit1 = unit1.loc[:, ['speech_id', 'speech']]
            comp_unit2 = unit2.loc[:, ['speech_id', 'speech']]
        else:
            comp_unit1 = unit1.loc[(unit1.party == parties[0]), ['speech_id', 'speech']]
            comp_unit2 = unit2.loc[(unit2.party == parties[1]), ['speech_id', 'speech']]

        hc, features = None, None
        try:
            hc, features = two_unit_test_topics(comp_unit1, comp_unit2, topics_df)
        except:
            continue

        # Write results to file
        outfile = None
        if on_cluster:
            outfile = os.path.expanduser('~/Data/results_topics{}_{}_{}_{}.csv'.format(topics_df.shape[1]-2, interval, parties[0], parties[1]))
        else:
            outfile = '../Data/results_topics{}_{}_{}_{}.csv'.format(topics_df.shape[1]-2, interval, parties[0], parties[1])
        unit_dates = [str(unit1.date[2])[:6], str(unit2.date[2])[:6]]
        with open(outfile, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            line = [j - i, unit_dates[0], parties[0], unit_dates[1], parties[1], hc, ','.join([str(f) for f in features])]
            writer.writerow(line)
