# %%
import pylab
import glob
import cv2
import os
import ntpath
import pims
# %%
Pathes = glob.glob('Video/*0054.mp4')
ImageFolder = 'Image'
# %%
for path in Pathes:
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    SavePath = os.path.join(ImageFolder, name)
    print(SavePath)
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    vid = cv2.VideoCapture(path)
    i = 0
    while(vid.isOpened()):
        ret,frame = vid.read()
        if ret == True:
            frameName = str(i)
            frameName = frameName.zfill(5)
            i = i + 1
            cv2.imwrite(os.path.join(SavePath, frameName + '.tiff'), frame)
        else:
            break
    vid.release()

# %% crop the image 

# %%
