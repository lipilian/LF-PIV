#%% import package∏
import numpy as np
from openpiv import tools, scaling, pyprocess, validation, process, filters
import matplotlib.pyplot as plt
import os
import sys
import cv2
from PIL import Image
import multiprocessing
from joblib import Parallel, delayed
import inspect
import glob
import plotly.figure_factory as ff
import matplotlib.gridspec as gridspec
import random
import tqdm
import matplotlib
#%% get all folder direction
calibrationFactor = 1
path = './Image/DSC_0054_subtrack_Rolling'
name = os.path.basename(path)
parPath = os.path.dirname(path)
SavePath = os.path.join(parPath, name + '_result')
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
imagenames = glob.glob(path + '\\*.tif')
fileNumber = len(imagenames)
N = fileNumber
print('fileNums is {}'.format(fileNumber))
fps = 30
startFrame = 932
DeltaFrame = 6
frame_a = tools.imread(imagenames[startFrame])
frame_b = tools.imread(imagenames[startFrame + DeltaFrame])
winsize = 20 # pixels
searchsize = 20#pixels
overlap = 10 # piexels
dt = DeltaFrame*1./fps # piexels
u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )
u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3)
u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=5, kernel_size=5)
u3, v3, mask1 = validation.local_median_val(u2,v2,3,3,1)
u4, v4 = filters.replace_outliers(u3, v3, method='localmean', max_iter=5, kernel_size=5)
mask1[:] = False
tools.save(x, y, u4, v4, mask1, os.path.join(parPath, 'test.txt'))

fig, ax = plt.subplots(2,1,figsize=(18,9))
ax[0].imshow(frame_a, cmap = plt.cm.gray, aspect = 'auto')

tools.display_vector_field(os.path.join(parPath, 'test.txt'),ax = ax[1], scale=2000, width=0.0005)
[m,n] = frame_a.shape
ax[1].set_xlim([0, n])
ax[1].set_ylim([m, 0])
fig.savefig(os.path.join(parPath, 'test1.png'))

# %% 
def process_node(i):
    winsize = 20 # pixels
    searchsize = 20 #pixels
    overlap = 10 # piexels
    dt = DeltaFrame*1./fps # piexels
    frame_a = tools.imread(imagenames[i])
    frame_b = tools.imread(imagenames[i+DeltaFrame])
    u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
    x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )
    u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )
    u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=5, kernel_size=5)
    u3, v3, mask1 = validation.local_median_val(u2,v2,3,3,1)
    u4, v4 = filters.replace_outliers(u3, v3, method='localmean', max_iter=5, kernel_size=5)
    mask1[:] = False
    tools.save(x, y, u4, v4, mask1, os.path.join(SavePath, str(i).zfill(5) + '.txt'))

    fig, ax = plt.subplots(2,1,figsize=(18,9))
    ax[0].imshow(frame_a, cmap = plt.cm.gray, aspect = 'auto')
    tools.display_vector_field(os.path.join(SavePath, str(i).zfill(5) + '.txt'),ax = ax[1], scale=2000, width=0.0005)
    [m,n] = frame_a.shape
    ax[1].set_xlim([0, n])
    ax[1].set_ylim([m, 0])
    fig.savefig(os.path.join(SavePath, str(i).zfill(5) + '.png'))

element_information = Parallel(n_jobs=6)(delayed(process_node)(node) for node in range(N - DeltaFrame))

#%% loop 4 cases
