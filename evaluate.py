# !/usr/bin/env python
"""
Script to compute pooled EER and min tDCF for ASVspoof2021 LA. 

Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has tje CM protocol and ASV score.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track

Example:
$: python evaluate.py score.txt ./keys eval
"""
import pandas
import pandas as pd
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ADD or ASV or EmoFake

def make_label_from_list(file_list):
    df = pd.DataFrame()
    name = []
    fake = []
    for i, line in file_list.iterrows():
        filename = line[0]
        name.append(filename)
        if filename[0] == 'S':
            fake.append('spoof')
        else:
            fake.append('bonafide')

    df[0] = name
    df[1] = fake
    return df


def CEER(cm_scores):
    y_true = []
    y_score = []

    for score, label in zip(cm_scores['1_x'], cm_scores['1_y']):

        if label == 'bonafide':
            y_true.append(1)
        else:
            y_true.append(0)
        y_score.append(score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print("eer:", eer, "thresh:", thresh)

    return eer


def make_label_of_ASV2021(cm_key_file):
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    name = []
    label = []
    for index, row in cm_data.iterrows():
        name.append(row[1])
        label.append(row[5])
    df = pd.DataFrame()
    df[0] = name
    df[1] = label
    return df


def make_label_of_ASV2019(cm_key_file):
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    name = []
    label = []
    for index, row in cm_data.iterrows():
        name.append(row[1])
        label.append(row[4])
    df = pd.DataFrame()
    df[0] = name
    df[1] = label
    return df


def make_label_of_ADD(cm_key_file):
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    name = []
    label = []
    for index, row in cm_data.iterrows():
        name.append(row[0].replace('.wav', ''))
        label.append(row[1].replace('fake', 'spoof').replace('genuine', 'bonafide'))
    df = pd.DataFrame()
    df[0] = name
    df[1] = label
    return df


def process_submission_scores(sb_sc):
    s = pd.DataFrame()
    a = []
    b = []
    for index, row in sb_sc.iterrows():
        name = row[0].replace('.wav', '').replace('.flac', '')
        a.append(name)
        score = row[1]
        b.append(score)
    s[0] = a
    s[1] = b
    return s


def eval_to_score_file(score_file, cm_key_file, db_name):
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    score = process_submission_scores(submission_scores)
    if db_name == 'EmoFake':
        cm_data = make_label_from_list(score)
    if db_name == 'ASV2021':
        cm_data = make_label_of_ASV2021(cm_key_file)
    if db_name == 'ASV2019':
        cm_data = make_label_of_ASV2019(cm_key_file)
    if db_name == 'ADD':
        cm_data = make_label_of_ADD(cm_key_file)

    cm_scores = score.merge(cm_data, left_on=0, right_on=0, how='inner')
    eer_cm = CEER(cm_scores)
    out_data = "eer(百分之):%.2f\n" % (100 * eer_cm)
    print(out_data, end="")
    return 100 * eer_cm


if __name__ == "__main__":
    submit_file = '/data6/zhaoyan/code/aasist-main/exp/emo_aasist_noleaf/ASV_64.txt'
    cm_key_file = '/data6/zhaoyan/data/ASVspoof2021_LA_eval/trial_metadata.txt'  # 不带后缀与地址
    databasename = 'ASV2021'
    _ = eval_to_score_file(submit_file, cm_key_file, databasename)
