# -*- coding: utf-8 -*-
"""w207_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZvaXtS9KDEEQglAlC6F4HuQIPCCvuH78
"""

import scipy.io
mat = scipy.io.loadmat('dataset_dreamer/DREAMER.mat')

import numpy as np

# import _pickle as cPickle
import pickle
import gzip
import csv

# DREAMER =
# struct with fields:
#  Data: {1×23 cell}
#  EEG_SamplingRate: 128
#  ECG_SamplingRate: 256
#  EEG_Electrodes: {'AF3' 'F7' 'F3' 'FC5' 'T7' 'P7' 'O1' 'O2' 'P8' 'T8' 'FC6' 'F4' 'F8' 'AF4'}
#  noOfSubjects: 23
#  noOfVideoSequences: 18
#  Disclaimer: 'While every care has been taken…'
#  Provider: 'University of the West of Scotland'
#  Version: '1.0.2'
#  Acknowledgement: 'The authors would like to thank…'
# DREAMER v1.0.2
# The cell DREAMER.Data{i} contains the data for the ith participant and is structured as follows:
# struct with fields:
#  Age: 'X'
#  Gender: 'X' ('male' or 'female')
#  EEG: [1×1 struct]
#  ECG: [1×1 struct]
#  ScoreValence: [18×1 double]
#  ScoreArousal: [18×1 double]
#  ScoreDominance: [18×1 double]
# ScoreValence, ScoreArousal and ScoreDominance are vectors that their ith element corresponds to the
# participant rating for the ith film clip in terms of Valence, Arousal, and Dominance respectively.
# The EEG and ECG recordings are stored in the DREAMER.Data{i}.EEG and DREAMER.Data{i}.ECG
# variables respectively which are structured as follows:
# struct with fields:
#  baseline: {18×1 cell}
#  stimuli: {18×1 cell}
# The recordings referring to the stimuli film clips are stored in the “stimuli” variable, while the
# recordings for the neutral clip shown before each film clip are stored in the “baseline” variable. The
# cells ....baseline{i} and ….stimuli{i} contain the data referring to the ith film clip.
# For ECG, each recording is in the form of an M x 2 matrix where M refers to the number of available
# samples and each column contains the sample of each of the two ECG channels.
# For EEG, each recording is in the form of an M x 14 matrix where M refers to the number of available
# samples and each column contains the sample of each of the 14 EEG channels.
# The jth column of the EEG recordings refers to the following electrode positions:
# j Position
# 1 AF3
# 2 F7
# 3 F3
# 4 FC5
# 5 T7
# 6 P7
# 7 O1
# 8 O2
# 9 P8
# 10 T8
# 11 FC6
# 12 F4
# 13 F8
# 14 AF4

print("keys:", mat.keys())

# open the file in the write mode
dp = open('out_dreamer/dreamer_participants.csv', 'w')


# create the csv writer
dp_writer = csv.writer(dp)
dp_writer.writerow(["participant_id", "age", "gender", "sample_id","score_valence", 
                    "score_arousal", "score_dominance"])


len(mat['DREAMER'][0,0])

# print("number of participants:", len(mat['DREAMER']['Data']))

print("number of participants:", len(mat['DREAMER']['Data'][0][0][0]))

participants_data = mat['DREAMER']['Data'][0][0][0]

#for i in [0,1]: #range(len(participants_data)):
for i in range(len(participants_data)):
  participant_csv_row_data = []

  p = participants_data[i]
  print("Participant:",i)
  print("age:", p['Age'][0][0][0])
  print("gender:", p['Gender'][0][0][0])

  participant_csv_row_data.append(i)
  participant_csv_row_data.append(p['Age'][0][0][0])
  participant_csv_row_data.append(p['Gender'][0][0][0])

  num_of_samples = len(p['EEG'][0][0][0][0]['baseline'])
  print("num_of_samples:", num_of_samples)
  #for j in [0]: #range(num_of_samples):
  for j in range(num_of_samples):
    print("Sample:",j)
    baseline = p['EEG'][0][0][0][0]['baseline'][j][0]
    stimuli = p['EEG'][0][0][0][0]['stimuli'][j][0]

    score_arousal = p['ScoreArousal'][0][0][j][0]
    score_valence = p['ScoreValence'][0][0][j][0]
    score_dominance = p['ScoreDominance'][0][0][j][0]

    #########################
    # write the participant to csv
    participant_csv_row_data.append(j)
    participant_csv_row_data.append(score_arousal)
    participant_csv_row_data.append(score_valence)
    participant_csv_row_data.append(score_dominance)
    dp_writer.writerow(participant_csv_row_data)

    #########################
    # write the stimuli data to csv
    dbd = open('out_dreamer/dreamer_p{:02d}_s{:02d}_a{:02d}_v{:02d}_d{:02d}.csv'.
               format(i,j,score_arousal,score_valence,score_dominance), 'w')
    dbd_writer = csv.writer(dbd)

    #append the headers of the 14 channels
    dbd_writer.writerow(["AF3",
                         "F7",
                         "F3",
                         "FC5",
                         "T7",
                         "P7",
                         "O1",
                         "O2",
                         "P8",
                         "T8",
                         "FC6",
                         "F4",
                         "F8",
                         "AF4"])
    
    #write stimuli data
    for s in stimuli:
      dbd_writer.writerow(s)
    
    dbd.close()

    #########################
    # write the baseline data to csv
    dbd = open('out_dreamer/dreamer_p{:02d}_s{:02d}_baseline.csv'.
               format(i,j), 'w')
    dbd_writer = csv.writer(dbd)

    #append the headers of the 14 channels
    dbd_writer.writerow(["AF3",
                         "F7",
                         "F3",
                         "FC5",
                         "T7",
                         "P7",
                         "O1",
                         "O2",
                         "P8",
                         "T8",
                         "FC6",
                         "F4",
                         "F8",
                         "AF4"])
   
    #write stimuli data
    for s in stimuli:
      dbd_writer.writerow(s)

    dbd.close()

    print("eeg baseline:", baseline[0].shape)
    print("eeg stimuli:", stimuli[0].shape)
    print("score_valence:", score_valence)
    print("---")


# close the file
dp.close()

