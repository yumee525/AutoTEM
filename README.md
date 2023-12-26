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































