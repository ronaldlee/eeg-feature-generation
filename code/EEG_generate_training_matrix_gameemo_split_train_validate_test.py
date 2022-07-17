#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2018:
    Original script by Dr. Luis Manso [lmanso], Aston University
    
2019, June:
    Revised, commented and updated by Dr. Felipe Campelo [fcampelo], Aston University
    (f.campelo@aston.ac.uk / fcampelo@gmail.com)
"""

import os, sys
import numpy as np
from EEG_feature_extraction import generate_feature_vectors_from_samples


def gen_training_matrix(directory_path, output_file, cols_to_ignore):
    """
    Reads the csv files in directory_path and assembles the training matrix with 
    the features extracted using the functions from EEG_feature_extraction.
    
    Parameters:
        directory_path (str): directory containing the CSV files to process.
        output_file (str): filename for the output file.
        cols_to_ignore (list): list of columns to ignore from the CSV

    Returns:
        numpy.ndarray: 2D matrix containing the data read from the CSV
    
    Author: 
        Original: [lmanso] 
        Updates and documentation: [fcampelo]
    """
    
    # Initialise return matrix
    FINAL_MATRIX = None
    
    participants_map = {}

    for x in os.listdir(directory_path):

        # Ignore non-CSV files
        if not x.lower().endswith('.csv'):
            continue
        
        # For safety we'll ignore files containing the substring "test". 
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in x.lower():
            continue
        try:
            #name, state, _ = x[:-4].split('-')

            participant = x[:3]
            state = x[3:5]

        except:
            print ('Wrong file name', x)
            sys.exit(-1)

        #based on Arousal-valence scale
        #1 - high arousal positive valence
        #2 - high arousal negative valence
        #3 - low arousal  negative valence
        #4 - low arousal  positive valence

        if state.lower() == 'g1':   #boring
            state = 3.
        elif state.lower() == 'g2': #calm
            state = 4.
        elif state.lower() == 'g3': #horror
            state = 2.
        elif state.lower() == 'g4': #funny
            state = 1.
        else:
            print ('Wrong file name', x)
            sys.exit(-1)
            
        print ('Using file', x)
        full_file_path = directory_path  +   '/'   + x
        vectors, header = generate_feature_vectors_from_samples(file_path = full_file_path, 
                                                                nsamples = 150, 
                                                                period = 1.,
                                                                state = state,
                                                                remove_redundant = True,
                                                                cols_to_ignore = cols_to_ignore)
        
        print ('resulting vector shape for the file', vectors.shape)
        
        if participants_map.get(participant) == None:
            participants_map[participant] = []

        participants_map[participant].append(vectors)
        
        ```
        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )
        ```    

    #print ('FINAL_MATRIX', FINAL_MATRIX.shape)
    
    keys =  list(participants_map.keys())
    keys = random.shuffle(keys)

    #get the first 60% as the training data
    number_of_participants = len(keys)
    number_of_training = int(number_of_participants * 0.6)

    validation_and_test = number_of_participants - number_of_training

    #split the rest in half for validation and test
    number_of_validation = int(validation_and_test * 0.5)
    number_of_test = number_of_participants - number_of_training - number_of_validation


    train_dataset = []
    for i in range(number_of_training):
        #participant data for the 4 tests
        participant_data = participants_map[keys[i]]
        for experiment_data in participant_data:
            train_dataset.append(experiment_data)

    np.savetxt("out_gameemo_train_dataset.csv", train_dataset, delimiter = ',',
            header = ','.join(header), 
            comments = '')

    validate_dataset = []
    for i in range(number_of_training, number_of_training+number_of_validation):
        participant_data = participants_map[keys[i]]
        for experiment_data in participant_data:
            validate_dataset.append(experiment_data)

    np.savetxt("out_gameemo_validate_dataset.csv", validate_dataset, delimiter = ',',
            header = ','.join(header), 
            comments = '')

    test_dataset = []
    for i in range(number_of_training+number_of_validation, number_of_participants):
        participant_data = participants_map[keys[i]]
        for experiment_data in participant_data:
            test_dataset.append(experiment_data)

    np.savetxt("out_gameemo_test_dataset.csv", test_dataset, delimiter = ',',
            header = ','.join(header), 
            comments = '')

    return None


if __name__ == '__main__':
    """
    Main function. The parameters for the script are the following:
        [1] directory_path: The directory where the script will look for the files to process.
        [2] output_file: The filename of the generated output file.
    
    ATTENTION: It will ignore the last column of the CSV file. 
    
    Author:
        Original by [lmanso]
        Documentation: [fcampelo]
"""
    if len(sys.argv) < 3:
        print ('arg1: input dir\narg2: output file')
        sys.exit(-1)
    directory_path = sys.argv[1]
    output_file = sys.argv[2]
    gen_training_matrix(directory_path, output_file, cols_to_ignore = None)
