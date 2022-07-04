import numpy as np
import scipy
import scipy.signal


matrix = np.array([
    [434.484,4346.153846153847,4043.076923076924],
    [434.584,4346.153846153847,4043.076923076924],
    [434.588,4371.794871794872,4075.384615384616],
    [434.592,4363.589743589744,4062.051282051282],
    ])

N   = matrix.shape[0]

fft = scipy.fft.fft(matrix, axis = 0)
print("fft:")
print(fft)

fft_values = np.abs(scipy.fft.fft(matrix, axis = 0))[0:N//2] * 2 / N

print(fft_values)
