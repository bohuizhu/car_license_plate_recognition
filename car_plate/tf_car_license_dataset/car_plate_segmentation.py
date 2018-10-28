# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import PIL
from PIL import Image
from resizeimage import resizeimage
import os
from PIL import Image
import PIL.ImageOps  


img = cv2.imread(r".\test_car_plate.png")  # read file

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # gray transform 
print ("step one finish ")

img_thre = img_gray

cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV, img_thre)

print ("step_two_finish")
#cv2.waitKey(0)

white = []  

black = [] 

height = img_thre.shape[0]

width = img_thre.shape[1]

white_max = 0

black_max = 0



# black groud and white digits 
for i in range(width):

    s = 0  

    t = 0 

    for j in range(height):

        if img_thre[j][i] == 0:

            s=s+1

        if img_thre[j][i] == 255:

            t=t+1

    white_max = max(white_max,s)

    black_max = max(black_max,t)

    white.append(s)

    black.append(t)

 
arg = False

def find_end(start_):

    end_ = start_+1

    for m in range(start_+1, width-1):

        if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max):  
            end_ = m
            break

    return end_
n = 1

start = 1

end = 2


l=[]

m=0
while n < width-2:

    n += 1

    if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
        start = n

        end = find_end(start)

        n = end

        if end-start > 5:
            cj = img_thre[1:height, start:end]
            
            l.append(cj)
SAVER_DIR="carplate_segmentation_digits/"
if not os.path.exists(SAVER_DIR):
    print ('do not save file')
    os.makedirs(SAVER_DIR)
    
            
for i in range(len(l)):
  # imshow(l[i])
    img = Image.fromarray(l[i])
    rate= img.size[1]/img.size[0]
   # print (rate)
   # img = PIL.ImageOps.invert(img)
    cover = resizeimage.resize_cover(img, [20,20*rate],validate=False)
    cover.save(r".\carplate_segmentation_digits\%s_digits.bmp" %i, img.format)
   