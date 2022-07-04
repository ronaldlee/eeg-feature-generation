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

# Commented since not used. [fcampelo]
# import sys
#from scipy.spatial.distance import euclidean

import numpy as np
import scipy
import scipy.signal

def matrix_from_csv_file(file_path):
    """
    Returns the data matrix given the path of a CSV file.
    
    Parameters:
        file_path (str): path for the CSV file with a time stamp in the first column
            and the signals in the subsequent ones.
            Time stamps are in seconds, with millisecond precision

    Returns:
        numpy.ndarray: 2D matrix containing the data read from the CSV
    
    Author: 
        Original: [lmanso] 
        Revision and documentation: [fcampelo]
    
    """
    
    csv_data = np.genfromtxt(file_path, delimiter = ',')
    full_matrix = csv_data[1:]
    headers = csv_data[0]
    
    return full_matrix, headers


def get_time_slice(full_matrix, start = 0., period = 1.):
    """
    Returns a slice of the given matrix, where start is the offset and period is 
    used to specify the length of the signal.
    
    Parameters:
        full_matrix (numpy.ndarray): matrix returned by matrix_from_csv()
        start (float): start point (in seconds after the beginning of records) 
        period (float): duration of the slice to be extracted (in seconds)

    Returns:
        numpy.ndarray: 2D matrix with the desired slice of the matrix
        float: actual length of the resulting time slice
        
    Author:
        Original: [lmanso]
        Reimplemented: [fcampelo]
    """
    
    # Changed for greater efficiency [fcampelo]
    rstart  = full_matrix[0, 0] + start
    index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
    index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))
    
    duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
    return full_matrix[index_0:index_1, :], duration


def feature_mean(matrix):
    """
    Returns the mean value of each signal for the full time window
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        
    Returns:
        numpy.ndarray: 1D array containing the means of each column from the input matrix
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    """
    
    ret = np.mean(matrix, axis = 0).flatten()
    names = ['mean_' + str(i) for i in range(matrix.shape[1])]
    return ret, names



def feature_mean_d(h1, h2):
    """
    Computes the change in the means (backward difference) of all signals 
    between the first and second half-windows, mean(h2) - mean(h1)
    
    Parameters:
        h1 (numpy.ndarray): 2D matrix containing the signals for the first 
        half-window
        h2 (numpy.ndarray): 2D matrix containing the signals for the second 
        half-window
        
    Returns:
        numpy.ndarray: 1D array containing the difference between the mean in h2 
        and the mean in h1 of all signals
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    
    """
    ret = (feature_mean(h2)[0] - feature_mean(h1)[0]).flatten()
    
    
    # Fixed naming [fcampelo]
    names = ['mean_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names



def feature_mean_q(q1, q2, q3, q4):
    """
    Computes the mean values of each signal for each quarter-window, plus the 
    paired differences of means of each signal for the quarter-windows, i.e.,
    feature_mean(q1), feature_mean(q2), feature_mean(q3), feature_mean(q4),
    (feature_mean(q1) - feature_mean(q2)), (feature_mean(q1) - feature_mean(q3)),
    ...
    
    Parameters:
        q1 (numpy.ndarray): 2D matrix containing the signals for the first 
        quarter-window
        q2 (numpy.ndarray): 2D matrix containing the signals for the second 
        quarter-window
        q3 (numpy.ndarray): 2D matrix containing the signals for the third 
        quarter-window
        q4 (numpy.ndarray): 2D matrix containing the signals for the fourth 
        quarter-window
        
    Returns:
        numpy.ndarray: 1D array containing the means of each signal in q1, q2, 
        q3 and q4; plus the paired differences of the means of each signal on 
        each quarter-window.
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    
    """
    v1 = feature_mean(q1)[0]
    v2 = feature_mean(q2)[0]
    v3 = feature_mean(q3)[0]
    v4 = feature_mean(q4)[0]
    ret = np.hstack([v1, v2, v3, v4, 
                     v1 - v2, v1 - v3, v1 - v4, 
                     v2 - v3, v2 - v4, v3 - v4]).flatten()
    
    
    # Fixed naming [fcampelo]
    names = []
    for i in range(4): # for all quarter-windows
        names.extend(['mean_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
    
    for i in range(3): # for quarter-windows 1-3
        for j in range((i + 1), 4): # and quarter-windows (i+1)-4
            names.extend(['mean_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
             
    return ret, names




def feature_stddev(matrix):
    """
    Computes the standard deviation of each signal for the full time window
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        
    Returns:
        numpy.ndarray: 1D array containing the standard deviation of each column 
        from the input matrix
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    """
    
    # fix ddof for finite sampling correction (N-1 instead of N in denominator)
    ret = np.std(matrix, axis = 0, ddof = 1).flatten()
    names = ['std_' + str(i) for i in range(matrix.shape[1])]
    
    return ret, names



def feature_stddev_d(h1, h2):
    """
    Computes the change in the standard deviations (backward difference) of all 
    signals between the first and second half-windows, std(h2) - std(h1)
    
    Parameters:
        h1 (numpy.ndarray): 2D matrix containing the signals for the first 
        half-window
        h2 (numpy.ndarray): 2D matrix containing the signals for the second 
        half-window
        
    Returns:
        numpy.ndarray: 1D array containing the difference between the stdev in h2 
        and the stdev in h1 of all signals
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    
    """
    
    ret = (feature_stddev(h2)[0] - feature_stddev(h1)[0]).flatten()
    
    # Fixed naming [fcampelo]
    names = ['std_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    
    return ret, names




def feature_moments(matrix):
    """
    Computes the 3rd and 4th standardised moments about the mean (i.e., skewness 
    and kurtosis) of each signal, for the full time window. Notice that 
    scipy.stats.moments() returns the CENTRAL moments, which need to be 
    standardised to compute skewness and kurtosis.
    Notice: Kurtosis is calculated as excess kurtosis, e.g., with the Gaussian 
    kurtosis set as the zero point (Fisher's definition)
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
    - https://en.wikipedia.org/wiki/Standardized_moment
    - http://www.econ.nyu.edu/user/ramseyj/textbook/pg93.99.pdf
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        
    Returns:
        numpy.ndarray: 1D array containing the skewness and kurtosis of each 
        column from the input matrix
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [fcampelo]
    """

    skw = scipy.stats.skew(matrix, axis = 0, bias = False)
    krt = scipy.stats.kurtosis(matrix, axis = 0, bias = False)
    ret  = np.append(skw, krt)
        
    names = ['skew_' + str(i) for i in range(matrix.shape[1])]
    names.extend(['kurt_' + str(i) for i in range(matrix.shape[1])])
    return ret, names




def feature_max(matrix):
    """
    Returns the maximum value of each signal for the full time window
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        
    Returns:
        numpy.ndarray: 1D array containing the max of each column from the input matrix
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    """
    
    ret = np.max(matrix, axis = 0).flatten()
    names = ['max_' + str(i) for i in range(matrix.shape[1])]
    return ret, names



def feature_max_d(h1, h2):
    """
    Computes the change in max values (backward difference) of all signals 
    between the first and second half-windows, max(h2) - max(h1)
    
    Parameters:
        h1 (numpy.ndarray): 2D matrix containing the signals for the first 
        half-window
        h2 (numpy.ndarray): 2D matrix containing the signals for the second 
        half-window
        
    Returns:
        numpy.ndarray: 1D array containing the difference between the max in h2 
        and the max in h1 of all signals
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    
    """
    
    ret = (feature_max(h2)[0] - feature_max(h1)[0]).flatten()
    
    # Fixed naming [fcampelo]
    names = ['max_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names


def feature_max_q(q1, q2, q3, q4):
    """
    Computes the max values of each signal for each quarter-window, plus the 
    paired differences of max values of each signal for the quarter-windows, 
    i.e., feature_max(q1), feature_max(q2), feature_max(q3), feature_max(q4),
    (feature_max(q1) - feature_max(q2)), (feature_max(q1) - feature_max(q3)),
    ...
    
    Parameters:
        q1 (numpy.ndarray): 2D matrix containing the signals for the first 
        quarter-window
        q2 (numpy.ndarray): 2D matrix containing the signals for the second 
        quarter-window
        q3 (numpy.ndarray): 2D matrix containing the signals for the third 
        quarter-window
        q4 (numpy.ndarray): 2D matrix containing the signals for the fourth 
        quarter-window
        
    Returns:
        numpy.ndarray: 1D array containing the max of each signal in q1, q2, 
        q3 and q4; plus the paired differences of the max values of each signal 
        on each quarter-window.
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    
    """
    v1 = feature_max(q1)[0]
    v2 = feature_max(q2)[0]
    v3 = feature_max(q3)[0]
    v4 = feature_max(q4)[0]
    ret = np.hstack([v1, v2, v3, v4, 
                     v1 - v2, v1 - v3, v1 - v4, 
                     v2 - v3, v2 - v4, v3 - v4]).flatten()
    
    
    # Fixed naming [fcampelo]
    names = []
    for i in range(4): # for all quarter-windows
        names.extend(['max_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
    
    for i in range(3): # for quarter-windows 1-3
        for j in range((i + 1), 4): # and quarter-windows (i+1)-4
            names.extend(['max_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
             
    return ret, names


def feature_min(matrix):
    """
    Returns the minimum value of each signal for the full time window
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        
    Returns:
        numpy.ndarray: 1D array containing the min of each column from the input matrix
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    """
    
    ret = np.min(matrix, axis = 0).flatten()
    names = ['min_' + str(i) for i in range(matrix.shape[1])]
    return ret, names



def feature_min_d(h1, h2):
    """
    Computes the change in min values (backward difference) of all signals 
    between the first and second half-windows, min(h2) - min(h1)
    
    Parameters:
        h1 (numpy.ndarray): 2D matrix containing the signals for the first 
        half-window
        h2 (numpy.ndarray): 2D matrix containing the signals for the second 
        half-window
        
    Returns:
        numpy.ndarray: 1D array containing the difference between the min in h2 
        and the min in h1 of all signals
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    
    """
    
    ret = (feature_min(h2)[0] - feature_min(h1)[0]).flatten()
    
    # Fixed naming [fcampelo]
    names = ['min_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names


def feature_min_q(q1, q2, q3, q4):
    """
    Computes the min values of each signal for each quarter-window, plus the 
    paired differences of min values of each signal for the quarter-windows, 
    i.e., feature_min(q1), feature_min(q2), feature_min(q3), feature_min(q4),
    (feature_min(q1) - feature_min(q2)), (feature_min(q1) - feature_min(q3)),
    ...
    
    Parameters:
        q1 (numpy.ndarray): 2D matrix containing the signals for the first 
        quarter-window
        q2 (numpy.ndarray): 2D matrix containing the signals for the second 
        quarter-window
        q3 (numpy.ndarray): 2D matrix containing the signals for the third 
        quarter-window
        q4 (numpy.ndarray): 2D matrix containing the signals for the fourth 
        quarter-window
        
    Returns:
        numpy.ndarray: 1D array containing the min of each signal in q1, q2, 
        q3 and q4; plus the paired differences of the min values of each signal 
        on each quarter-window.
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    
    """
    v1 = feature_min(q1)[0]
    v2 = feature_min(q2)[0]
    v3 = feature_min(q3)[0]
    v4 = feature_min(q4)[0]
    ret = np.hstack([v1, v2, v3, v4, 
                     v1 - v2, v1 - v3, v1 - v4, 
                     v2 - v3, v2 - v4, v3 - v4]).flatten()
    
    
    # Fixed naming [fcampelo]
    names = []
    for i in range(4): # for all quarter-windows
        names.extend(['min_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
    
    for i in range(3): # for quarter-windows 1-3
        for j in range((i + 1), 4): # and quarter-windows (i+1)-4
            names.extend(['min_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
             
    return ret, names


def feature_covariance_matrix(matrix):
    """
    Computes the elements of the covariance matrix of the signals. Since the 
    covariance matrix is symmetric, only the lower triangular elements 
    (including the main diagonal elements, i.e., the variances of eash signal) 
    are returned. 
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        
    Returns:
        numpy.ndarray: 1D array containing the variances and covariances of the 
        signals
        list: list containing feature names for the quantities calculated.
        numpy.ndarray: 2D array containing the actual covariance matrix

    Author:
        Original: [fcampelo]
    """
    
    #find covariance of the given matrix
    covM = np.cov(matrix.T)
    indx = np.triu_indices(covM.shape[0])

    #ret is a lower-triangular matrix
    ret  = covM[indx]

    #flatten the matrix and only keep values in the lower-triangle 
    names = []
    for i in np.arange(0, covM.shape[1]):
        for j in np.arange(i, covM.shape[1]):
            names.extend(['covM_' + str(i) + '_' + str(j)])
    
    return ret, names, covM


def feature_eigenvalues(covM):
    """
    Computes the eigenvalues of the covariance matrix passed as the function 
    argument.
    
    Parameters:
        covM (numpy.ndarray): 2D [nsignals x nsignals] covariance matrix of the 
        signals in a time window
        
    Returns:
        numpy.ndarray: 1D array containing the eigenvalues of the covariance 
        matrix
        list: list containing feature names for the quantities calculated.

    Author:
        Original: [lmanso]
        Revision and documentation: [fcampelo]
    """
    
    ret   = np.linalg.eigvals(covM).flatten()
    names = ['eigenval_' + str(i) for i in range(covM.shape[0])]
    return ret, names


def feature_logcov(covM):
    """
    Computes the matrix logarithm of the covariance matrix of the signals. 
    Since the matrix is symmetric, only the lower triangular elements 
    (including the main diagonal) are returned. 
    
    In the unlikely case that the matrix logarithm contains complex values the 
    vector of features returned will contain the magnitude of each component 
    (the covariance matrix returned will be in its original form). Complex 
    values should not happen, as the covariance matrix is always symmetric 
    and positive semi-definite, but the guarantee of real-valued features is in 
    place anyway. 
    
    Details:
        The matrix logarithm is defined as the inverse of the matrix 
        exponential. For a matrix B, the matrix exponential is
        
            $ exp(B) = \sum_{r=0}^{\inf} B^r / r! $,
        
        with 
        
            $ B^r = \prod_{i=1}^{r} B / r $.
            
        If covM = exp(B), then B is a matrix logarithm of covM.
    
    Parameters:
        covM (numpy.ndarray): 2D [nsignals x nsignals] covariance matrix of the 
        signals in a time window
        
    Returns:
        numpy.ndarray: 1D array containing the elements of the upper triangular 
        (incl. main diagonal) of the matrix logarithm of the covariance matrix.
        list: list containing feature names for the quantities calculated.
        numpy.ndarray: 2D array containing the matrix logarithm of covM
        

    Author:
        Original: [fcampelo]
    """
    log_cov = scipy.linalg.logm(covM)
    indx = np.triu_indices(log_cov.shape[0])
    ret  = np.abs(log_cov[indx])
    
    names = []
    for i in np.arange(0, log_cov.shape[1]):
        for j in np.arange(i, log_cov.shape[1]):
            names.extend(['logcovM_' + str(i) + '_' + str(j)])
    
    return ret, names, log_cov



def feature_fft(matrix, period = 1., mains_f = 50., 
                filter_mains = True, filter_DC = True,
                normalise_signals = True,
                ntop = 10, get_power_spectrum = True):
    """
    Computes the FFT of each signal. 
    
    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
        values of nsignals for a time window of length nsamples
        period (float): width (in seconds) of the time window represented by
        matrix
        mains_f (float): the frequency of mains power supply, in Hz.
        filter_mains (bool): should the mains frequency (plus/minus 1Hz) be 
        filtered out?
        filter_DC (bool): should the DC component be removed?
        normalise_signals (bool): should the signals be normalised to the 
        before interval [-1, 1] before computing the FFT?
        ntop (int): how many of the "top N" most energetic frequencies should 
        also be returned (in terms of the value of the frequency, not the power)
        get_power_spectrum (bool): should the full power spectrum of each 
        signal be returned (in terms of magnitude of each frequency component)
        
    Returns:
        numpy.ndarray: 1D array containing the ntop highest-power frequencies 
        for each signal, plus (if get_power_spectrum is True) the magnitude of 
        each frequency component, for all signals.
        list: list containing feature names for the quantities calculated. The 
        names associated with the power spectrum indicate the frequencies down 
        to 1 decimal place.

    Author:
        Original: [fcampelo]
    """
    
    # Signal properties
    N   = matrix.shape[0] # number of samples
    T = period / N        # Sampling period
    
    # Scale all signals to interval [-1, 1] (if requested)
    if normalise_signals:
        matrix = -1 + 2 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    
    # Compute the (absolute values of the) FFT
    # Extract only the first half of each FFT vector, since all the information
    # is contained there (by construction the FFT returns a symmetric vector).
    fft_values = np.abs(scipy.fft.fft(matrix, axis = 0))[0:N//2] * 2 / N
    
    # Compute the corresponding frequencies of the FFT components
    freqs = np.linspace(0.0, 1.0 / (2.0 * T), N//2)
    
    # Remove DC component (if requested)
    if filter_DC:
        fft_values = fft_values[1:]
        freqs = freqs[1:]
        
    # Remove mains frequency component(s) (if requested)
    if filter_mains:
        indx = np.where(np.abs(freqs - mains_f) <= 1)
        fft_values = np.delete(fft_values, indx, axis = 0)
        freqs = np.delete(freqs, indx)
    
    # Extract top N frequencies for each signal
    indx = np.argsort(fft_values, axis = 0)[::-1]
    indx = indx[:ntop]
    
    # flatten in column-major (https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)
    # a = np.array([[1,2], [3,4]])
    # a.flatten('F')
    # array([1, 3, 2, 4])
    # for 'topFreq_' below
    ret = freqs[indx].flatten(order = 'F')
    
    # Make feature names
    names = []
    for i in np.arange(fft_values.shape[1]):
        names.extend(['topFreq_' + str(j) + "_" + str(i) for j in np.arange(1,11)])
    
    if (get_power_spectrum):
        ret = np.hstack([ret, fft_values.flatten(order = 'F')])
        
        for i in np.arange(fft_values.shape[1]):
            names.extend(['freq_' + "{:03d}".format(int(j)) + "_" + str(i) for j in 10 * np.round(freqs, 1)])
    
    return ret, names


"""
Returns a number of feature vectors from a labeled CSV file, and a CSV header 
corresponding to the features generated.
full_file_path: The path of the file to be read
samples: size of the resampled vector
period: period of the time used to compute feature vectors
state: label for the feature vector
"""
def generate_feature_vectors_from_samples(file_path, nsamples, period, state): 
    # Read the matrix from file
    matrix, headers = matrix_from_csv_file(file_path)

    headers = np.append(headers,"Label")
    
    # We will start at the very begining of the file
    t = 0.
    
    # No previous vector is available at the start
    previous_vector = None
    
    # Initialise empty return object
    ret = None
    
    # Until an exception is raised or a stop condition is met
    while True:
        # Get the next slice from the file (starting at time 't', with a 
        # duration of 'period'
        # If an exception is raised or the slice is not as long as we expected, 
        # return the current data available
        try:
            s, dur = get_time_slice(matrix, start = t, period = period)
        except IndexError:
            break
        if len(s) == 0:
            break
        if dur < 0.9 * period:
            break
        
        # Perform the resampling of the vector
        # This is to 'smooth out' the data, and also standardize to the same number of data points (nsamples). 
        # The 'time' might be irregular when collected, so resample just to make the time at regular period.
        ry, rx = scipy.signal.resample(s[:, 1:], num = nsamples, 
                                 t = s[:, 0], axis = 0)
        
        # Slide the slice by 1/2 period
        t += 0.5 * period
        
        print("RRRR rx.shape:",rx.shape)
        print("RRRR ry.shape:",ry.shape)

        states = np.repeat(3, rx.shape[0])
        print("RRRR states.shape:",states.shape)

        feature_vector = np.vstack((rx, ry, states))
        print("RRRR feature_vector.shape:", feature_vector.shape)
        
        if ret is None:
            ret = feature_vector
        else:
            ret = np.vstack([ret, feature_vector])

    # Return
    return ret, headers 



# ========================================================================
