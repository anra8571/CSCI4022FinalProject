from os.path import dirname, join as pjoin
import scipy.io as sio

mat_content = sio.loadmat("fNIRS 04.mat")
print(mat_content)