# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

import os
from os.path import dirname, join as pjoin
import scipy.io as sio # type: ignore
import numpy as np # type: ignore

# Vars to set manually
filename = "4_feature_vector.mat"
outfile = "4_fv.np"

# Loads MATLAB file into a dictionary
cwd = os.getcwd()
directory = f"{cwd}/{filename}"
print("File directory: ", directory)
mat_content = sio.loadmat(directory)
print("Data: ", mat_content['data'])

# This is how to access the long list
print("Length of Data: ", len(mat_content['data'][0]))

# i = trial
# j = channel
# k = epoch

data = []
for i in range(75):
    channel = []
    for j in range(40):
        epoch = []
        for k in range(3):
            # print(f"i: {i}, j: {j}, k: {k}, total: {(40 * i) + (3 * j) + (k)}")
            epoch.append(mat_content['data'][0][(40 * i) + (3 * j) + (k)])
        channel.append(epoch)
    data.append(channel)

print("Length of 1st Dimension: ", len(data))
print("Length of 2nd Dimension: ", len(data[0]))
print("Length of 3rd Dimension: ", len(data[0][0]))
with open(outfile, 'wb') as f:
    np.save(f, np.array(data))