import numpy as np

time = np.array([1,2,3])

features = np.array([
        [ "a", "b", "c", "Y"],
        [ "d", "e", "f", "N"],
        [ "g", "h", "i", "Y"]
        ])

print("time.shape:",time.shape)
print("features.shape:",features.shape)

time = time[..., None]
print("time.shape after:",time.shape)


print("time:")
print(time)
print("features:") 
print(features)

v = np.hstack((time, features))

print("v:")
print(v)
