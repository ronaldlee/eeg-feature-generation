import numpy as np

data = np.array([
[1,2,3],
[4,5,6],
[7,8,9],
[10,11,10],
[9,1,5]
        ])

covM = np.cov(data.T)


print(covM)
