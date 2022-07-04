import numpy as np
import scipy
import scipy.signal

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
    print("rstart:",rstart)
    print("period:",period)
    print("rstart + period:",rstart+period)

    print("index_1:", np.where(full_matrix[:, 0] <= rstart + period))
    print("index_1 np.max1:", np.max(np.where(full_matrix[:, 0] <= rstart + period)))
    index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
    index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))

    duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
    return full_matrix[index_0:index_1, :], duration

s = np.array([
    [434.484,4346.153846153847,4043.076923076924],
    [434.584,4346.153846153847,4043.076923076924],
    [434.588,4371.794871794872,4075.384615384616],
    [434.592,4363.589743589744,4062.051282051282],
    [434.596,4327.179487179487,4024.102564102565],
    [434.602,4358.974358974359,4048.205128205129],
    [434.608,4369.743589743591,4036.9230769230776],
    [434.702,4358.974358974359,4048.205128205129],
    [434.827,4358.974358974359,4048.205128205129],
    [435.008,4369.743589743591,4036.9230769230776],
    [435.009,4379.743589743591,4036.9230769230776],
    [435.010,4380.743589743591,4036.9230769230776],
    [435.658,4369.743589743591,4036.9230769230776],
    [435.772,4358.974358974359,4048.205128205129],
    [435.897,4358.974358974359,4048.205128205129],
    [436.003,4369.743589743591,4036.9230769230776],
    [436.114,4369.743589743591,4036.9230769230776],
    ])


window = get_time_slice(s)

print("window:")
print(window)

