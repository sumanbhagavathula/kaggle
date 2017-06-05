from io import BytesIO
from matplotlib.pyplot import imshow
from IPython.display import display
import copy
import numpy as np
from PIL import Image, ImageOps  # PIL is the Python Imaging Library
import requests
import imageaugmentationfunctions as imgaug
import utilityfunctions as util

imgtype = 'test'
#images_dir = '/home/ubuntu/src/tensorflow/tensorflow/models/image/imagenet/TUTORIAL_DIR/images/'
imagedir = r'C:\Users\sumabh\OneDrive\MLDS\UW-MSDS\DATA558\GitHub\kaggle\data'
if imgtype == 'train':
    augmentedimagedir = r'C:\Users\sumabh\desktop\augimages'
if imgtype == 'test':
    augmentedimagedir = r'C:\Users\sumabh\desktop\augimages'

imgfilelist = util.list_all_files(imagedir,imgtype)

for imgfile in imgfilelist:
    img = imgaug.image_open(imgfile)
    id_img = imgaug.identity_image(img)
    imgaug.image_save(augmentedimagedir, img)
    break
