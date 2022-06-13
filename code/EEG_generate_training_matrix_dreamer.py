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
    
    for x in os.listdir(directory_path):

        # Ignore non-CSV files
        if not x.lower().endswith('.csv'):
            continue
        
        # For safety we'll ignore files containing the substring "test". 
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in x.lower():
            continue

        if not ('dreamer_p00_s00_a03_v04_d02.csv' in x.lower()):
            continue

        if 'baseline' in x.lower():
            continue

        print("RRR find file")

        try:
            #name format: dreamer_p11_s08_a04_v01_d04.csv

            #arousal/valence/domainance in scale 1-5
            #low/high arousal (calm/excited), 
            #low/high valence (unpleasant/pleasant), and low/high dominance (without control/empowered)
            name, participant, sample, arousal, valence, domainance = x[:-4].split('_')

            
            arousal = int(arousal[1:]) 
            valence = int(valence[1:]) 
            domainance= int(domainance[1:]) 

        except:
            print ('Wrong file name', x)
            sys.exit(-1)

        #based on Arousal-valence scale
        #1 - high arousal positive valence
        #2 - high arousal negative valence
        #3 - low arousal  negative valence
        #4 - low arousal  positive valence

        if arousal > 3 and valence > 3:
            state = 1.
        elif arousal > 3 and valence <= 3:
            state = 2.
        elif arousal <= 3 and valence <= 3:
            state = 3.
        elif arousal <= 3 and valence > 3:
            state = 4.
        else:
            print ('Wrong file name', x)
            sys.exit(-1)
            
        print("arousal:", arousal, "; valence:", valence, "; state:", state)

        print ('Using file', x)
        full_file_path = directory_path  +   '/'   + x
        vectors, header = generate_feature_vectors_from_samples(file_path = full_file_path, 
                                                                nsamples = 150, 
                                                                period = 1.,
                                                                state = state,
                                                                remove_redundant = True,
                                                                cols_to_ignore = cols_to_ignore)
        
        print ('resulting vector shape for the file', vectors.shape)
        
        
        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )

    print ('FINAL_MATRIX', FINAL_MATRIX.shape)
    
    # Shuffle rows
    np.random.shuffle(FINAL_MATRIX)
    
    # Save to file
    np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
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
    gen_training_matrix(directory_path, output_file, cols_to_ignore = -1)
