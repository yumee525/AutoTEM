import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import sys
import openpyxl

## -----------------------------------------------------------
## Parameters
## -----------------------------------------------------------
## Flages
flag_inv = 0    ## Invert black/white (0/1 --> N/Y)
flag_norm = 1   ## Normalize gray scale to (0 ~ 255)
flag_matplt = 1 ## Matlibplot (0/1 --> N/Y)
flag_blur = 1   ## Blurry methods (0/1/2 --> no/Average/Gaussian)
flag_Wedg = 1   ## White edge (0/1 --> N/Y)
flag_win = 0    ## Interactive window (0/1 --> N/Y)
flag_autoM = 1  ## Auto measurement (0/1 --> N/Y)
flag_info = 0   ## Information level (0/1/2 --> necessary/details/all)
flag_itype = 0  ## Image type (0/1 --> jpg/png)
flag_transf = 1 ## Transformation before/after Canny (0/1 --> before/after)
flag_Qs = 0     ## Questions for input (0/1 --> No/Yes)

## Define input/output name(定義輸入/輸出名稱)
inputname = "Ref-4"
if flag_itype == 0:
    pathname = "./"+inputname+".jpg"
if flag_itype == 1:
    pathname = "./"+inputname+".png"

## Auto extract
W_box_nm = 30
H_box_nm = 80
delta_nm = 1
auto_name = "Final extract"

## Reference for image scaling(影像縮放參考)
P_ref = 80  ## [#] Ref. pixels
L_ref = 10  ## [nm] Ref. length
nm_pix_ref = L_ref/P_ref
pix_nm_ref = int(P_ref/L_ref)

## Normalization ranges(標準化範圍)
min_l = 0
max_l = 255

## Interactive window(互動視窗)
window_name = "Canny Method"
title_track = "max threshold"
title_track2 = "min/max ratio"

## Blurry kernal(模糊內核)
n_blur = 4           ## [-]  Number of blurring cycles
blur_level = 5       ## [-]  Number of pixels for blurring (e.g. 5 means 5*5 pixels)

## Extract threshold(提取閾值)
max_th_fix = 129      ## [-]  Available only if flag_Qs == 0
ratio_fix = 45       ## [-]  Available only if flag_Qs == 0

## Number of plugs to be extracted
n_plugs_fix = 4      ## [-]  Available only if flag_Qs == 0
L_scale_bar_fix = 20 ## [nm] Available only if flag_Qs == 0

## Visualization parameters(可視化參數)
## w_arbitrary = 500  ## [pixels] Arbitrary number
w_arbitrary = 800  ## [pixels] Arbitrary number

## Image type
if flag_itype == 0:
    img_quality = [cv.IMWRITE_JPEG_QUALITY, 100]  ## Quality 0~100
if flag_itype == 1:
    img_quality = [cv.IMWRITE_PNG_COMPRESSION, 0] ## Compress level 0~5

## -----------------------------------------------------------
## Log file
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Program start ##\n")

print("Flags:")
print("flag_inv\t",flag_inv)
print("flag_matplt\t",flag_matplt)
print("flag_norm\t",flag_norm)
print("flag_blur\t",flag_blur)
print("flag_Wedg\t",flag_Wedg)
print("flag_win\t",flag_win,"\n")

print("Grayscale ranges:")
print("max_l\t",max_l)
print("min_l\t",min_l)

print("\nFiles:")
print("Input file path:\t",pathname)

## -----------------------------------------------------------
## Define functions
## -----------------------------------------------------------
## Scaling image by keeping aspect ratio(固定比例縮放影像)
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA)

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter), r

## Find groups of a list of numbers & output start/end index in the list(尋找數字清單的群組並輸出清單中的開始/結束索引)
def myislandinfo(y, trigger_val, stopind_inclusive=True):

    ## Setup "sentients" on either sides to make sure we have setup
    ## "ramps" to catch the start and stop for the edge islands
    ## (left-most and right-most islands) respectively
    y_ext = np.r_[False,y != trigger_val, False]

    ## Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    ## Lengths of islands if needed
    lens = idx[1::2]-idx[:-1:2]

    ## Using a stepsize of 2 would get us start and stop indices for each island
    return list(zip(idx[:-1:2], idx[1::2]-int(stopind_inclusive))), lens

## Track mouse positions & do something based on actions of the mouse
def drawCross(event, x, y, flags, param):

    image_temp = param[0]
    image_name = param[1]

    if event == cv.EVENT_MOUSEMOVE:
        imgCopy = image_temp.copy()
        cv.line(imgCopy, (x-10000,y), (x+10000,y), (0, 0, 255), 1)
        cv.line(imgCopy, (x,y-10000), (x,y+10000), (0, 0, 255), 1)
        cv.line(imgCopy, (x-10,y), (x+10,y), (0, 255, 0), 1)
        cv.line(imgCopy, (x,y-10), (x,y+10), (0, 255, 0), 1)
        cv.imshow(image_name, imgCopy)

    if event == cv.EVENT_LBUTTONDBLCLK:
        print("Selected (x,y) = (",x,",",y,")")
        cv.line(image_temp, (x-10,y), (x+10,y), (77, 208, 225), 1)
        cv.line(image_temp, (x,y-10), (x,y+10), (77, 208, 225), 1)
        cv.imshow(image_name, image_temp)

        coords.append([x, y])

## Default OpenCV rotation (cropping image outside the frame)
def DefaultRot(rotateImage, angle):

    print("\nCall default rotation function:")

    ## Get image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]
    print("img_s W/H =",imgWidth,",",imgHeight)

    ## Image center coordinate (x,y)=(w,h)
    centerX, centerY = imgWidth//2, imgHeight//2
    print("center_old =",centerX,",",centerY)

    ## Define 2D rotation matrix
    rotationMatrix = cv.getRotationMatrix2D((centerX, centerY), angle, 1.0)
  
    ## Perform rotation
    rotatingimage = cv.warpAffine(rotateImage, rotationMatrix, (imgWidth, imgHeight))

    return rotatingimage
    
## Modified OpenCV rotation without cropping/cutting sides
def ModifiedRot(rotateImage, angle):

    print("Call modified rotation function:")
    
    ## Get image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]
    print("img_s W/H =",imgWidth,",",imgHeight)

 ## Image center coordinate (x,y)=(w,h)
    centerX, centerY = imgWidth//2, imgHeight//2
    print("center_old =",centerX,",",centerY)

## 2D rotation matrix
    rotationMatrix = cv.getRotationMatrix2D((centerX, centerY), angle, 1.0)
    print("M_rot_old",rotationMatrix)

## Take out sin and cos values from rotationMatrix
## Use numpy absolute function to make positive value
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])
    print("cos,sin =",cosofRotationMatrix,",",sinofRotationMatrix)
    
## Compute new height & width of an image for warpAffine function to prevent cropping images
    newImageWidth = int((imgWidth * cosofRotationMatrix) + (imgHeight * sinofRotationMatrix))
    newImageHeight = int((imgWidth * sinofRotationMatrix) + (imgHeight * cosofRotationMatrix))
    print("img_s rescale W/H =",newImageWidth,",",newImageHeight)
    print("center_new =",newImageWidth//2,",",newImageHeight//2)

## Update the values of rotation matrix (delta=new_center-old_center)
    rotationMatrix[0][2] += (newImageWidth/2) - centerX
    rotationMatrix[1][2] += (newImageHeight/2) - centerY
    print("M_rot_new",rotationMatrix,"\n")

## Perform rotation
    rotatingimage = cv.warpAffine(rotateImage,rotationMatrix,(newImageWidth,newImageHeight),cv.INTER_AREA)
  
    return rotatingimage

## -----------------------------------------------------------
## Load image
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Load image ##\n")

## Load image
img = cv.imread(pathname,cv.IMREAD_UNCHANGED)

(h_ori, w_ori) = img.shape[:2]
print("Original image (H/W) =",h_ori,"x",w_ori)












