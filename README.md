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
n_plugs_fix = 4      
## [-]  Available only if flag_Qs == 0
L_scale_bar_fix = 20 
## [nm] Available only if flag_Qs == 0

## Visualization parameters(可視化參數)
## w_arbitrary = 500
## [pixels] Arbitrary number
w_arbitrary = 800  
## [pixels] Arbitrary number

## Image type
if flag_itype == 0:
    img_quality = [cv.IMWRITE_JPEG_QUALITY, 100]  
## Quality 0~100
if flag_itype == 1:
    img_quality = [cv.IMWRITE_PNG_COMPRESSION, 0] 
## Compress level 0~5

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

## Change format
print("img.shape",img.shape)
if(len(img.shape) >= 3):
    if(img.shape[2] >= 3):
        print("channel img.shape[2]=",img.shape[2])
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

## -----------------------------------------------------------
## Invert or normalize grayscale images(反轉與正規化影像)
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Invert / normalize grayscale images ##\n")

## Invert image(反轉影像)
if flag_inv == 1:
    img = cv.bitwise_not(img)
    cv.imshow("Invert black/white", img)
    cv.waitKey(1)

## Normalization(正規化影像)
if flag_norm == 1:
    img = cv.normalize(img, None, min_l, max_l, cv.NORM_MINMAX) 
    if flag_info >= 1:
        cv.imshow("Normalized image", img)
        cv.waitKey(1)

## -----------------------------------------------------------
## Find scaling & rotation factors for a dividable value of pixels/nm
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Find scaling & rotation factors ##\n")

## Scale original image w/ an arbitrary ratio (Preview)
img_rgb_arb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
img_rgb_arb, r_resize_arb = ResizeWithAspectRatio(img_rgb_arb, width=w_arbitrary)
cv.imshow("Image preview", img_rgb_arb) 

print("Please double click left mouse to select start & end positions of the scale bar.\n") 
## --- debug start --- Replace by print
print("Please select 2 points at the same interface for determining the rotation angle.\n")

## Find scale bar(找尋比例尺)
coords = []
input_param =[img_rgb_arb,"Image preview"]
cv.setMouseCallback("Image preview", drawCross, input_param)
cv.waitKey(0)

if flag_Qs == 1:
    try:
        L_ab = float(input("Scale bar length [nm]: "))
    except ValueError:
        print("Not a number")
else:
    L_ab = L_scale_bar_fix        
print("Scale bar used [nm]:",L_ab,"\n")

coords = np.reshape(coords,(4,2))

## Calculate transformation factors
print("\n-----------------------------------------------------------")
print("## Calculate transfromation factors ##\n")

P_a_x = coords[0,0]
P_a_y = coords[0,1]
P_b_x = coords[1,0]
P_b_y = coords[1,1]
P_ab = math.sqrt((P_b_x-P_a_x)*(P_b_x-P_a_x)+(P_b_y-P_a_y)*(P_b_y-P_a_y))

nm_pix_ab = L_ab/P_ab       ## nm/pixel
pix_nm_ab = P_ab/L_ab       ## pixel/nm

r_ref_ab = pix_nm_ref/pix_nm_ab
pix_nm_check = pix_nm_ab*r_ref_ab 

## Target scaled width
w_target = int(w_ori*r_ref_ab*r_resize_arb)

print("P_ab:\t",P_ab)
print("nm_pix_ref:\t",format(nm_pix_ref,".3f"),"\tnm/pixel ; pix_nm_ref:\t",format(pix_nm_ref,".3f"),"\tpixel/nm")
print("nm_pix_ab:\t",format(nm_pix_ab,".3f"),"\tnm/pixel ; pix_nm_ab:\t",format(pix_nm_ab,".3f"),"\tpixel/nm")
print("r_ref_ab:\t",format(r_ref_ab,".3f"))
print("reference pix_nm:\t",pix_nm_ref,"pixel/nm")
print("r_resize_arb:\t",format(r_resize_arb,".3f"),"\n")

print("pix_nm_check (pix_nm_ab*r_ref_ab):\t",format(pix_nm_check,".3f"),"pixel/nm (expect = pix_nm_ref)")
print("w_target (w_ori*r_ref_ab*r_resize_arb):\t",format(w_target,".3f"),"pixels\n")

## Find 2D rotation angle & define 2D rotation matrix
P_a2_x = coords[2,0]
P_a2_y = coords[2,1]
P_b2_x = coords[3,0]
P_b2_y = coords[3,1]

ddx = P_b2_x-P_a2_x
ddy = -(P_b2_y-P_a2_y)

print("ddx=",ddx)
print("ddx=",ddy,"\n")

if ddx == 0 and ddy > 0:
    angle_deg = 90
    angle_rad = np.radians(angle_deg)
elif ddx == 0 and ddy < 0:
    angle_deg = 270
    angle_rad = np.radians(angle_deg)
elif ddy == 0:
    angle_rad = 0.0
    angle_deg = 0.0
else:
    angle_rad = np.arctan(ddy/abs(ddx))
    if ddx >= 0:
        angle_deg = np.degrees(angle_rad)
    else:
        angle_deg = 180-np.degrees(angle_rad)

print("angle(rad) =",angle_rad)
print("angle(deg) =",angle_deg)

## -----------------------------------------------------------
## Perform transformation before Canny(Canny前進行轉換)
## -----------------------------------------------------------
if flag_transf == 0:
    print("\n-----------------------------------------------------------")
    print("## Perform transformation before Canny ##\n")

    img_x = ResizeWithAspectRatio(img, width=w_target)[0]

## 根據"寬度"調整大小
    (h_x, w_x) = img_x.shape[:2]
    print("Resize bf. Canny (H/W) =",h_x,"x",w_x)

    cv.imshow("Resized bf. Canny",img_x)
## 顯示影像，只執行這段命令可能會造成視窗崩潰或沒有反應，須再執行cv.waitKey()
    cv.waitKey(1)
## 等待與讀取使用者所按下的按鍵，等待時間預設單位為毫秒

## -----------------------------------------------------------
## Interactive Canny edge extraction on "original" image & resize for visualization(對「原始」影像進行互動式 Canny 邊緣擷取並調整大小以實現視覺化)
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Interactive Canny extraction  ##\n")

if flag_win == 1:
    def CannyThreshold(val):
        max_th = cv.getTrackbarPos(title_track, window_name)

        ratio  = cv.getTrackbarPos(title_track2, window_name)/100

        ## Pre-blurry before Canny
        img_previous = img
        for it in range(1,n_blur+1):
            if it == 1:
                img_in = img
            else:
                img_in = img_previous
        
            if flag_blur == 0:
                img_blur = img_in
            elif flag_blur == 1:
                img_blur = cv.blur(img_in, (blur_level,blur_level))
            elif flag_blur == 2:
                img_blur = cv.GaussianBlur(img_in, (blur_level,blur_level), 0)

            img_previous = img_blur

        ## Canny extract
        detected_edges = cv.Canny(img_blur, max_th*ratio, max_th, blur_level)

        ## Define mask & apply mask
        mask = detected_edges != 0

        if flag_Wedg == 1:
            detected_edges[mask] = 255
            dst = detected_edges
        else:
            dst = img * (mask[:,:].astype(img.dtype))

        ## Resize af. edge extraction for better visualization
        dst_s = ResizeWithAspectRatio(dst, width=w_arbitrary)[0]
        cv.imshow(window_name, dst_s)

    ## Use defined function
    cv.namedWindow(window_name)
    cv.createTrackbar(title_track, window_name , min_l, max_l, CannyThreshold)
    cv.createTrackbar(title_track2, window_name , 0, 100, CannyThreshold)

    CannyThreshold(0)
    cv.waitKey(0)

## Get value from user
if flag_Qs == 1:
    try:
        max_th = float(input("Please enter max threshold [0-255]:"))
    except ValueError:
        print("Not a number")
    try:
        ratio = float(input("Please enter min/max ratio [0-100]:"))
    except ValueError:
        print("Not a number")
else:
    max_th = max_th_fix
    ratio = ratio_fix

ratio = ratio/100
print("Entered (max_th,ratio) = (",max_th,",",ratio,")\n")
min_th = max_th*ratio

## Define input/output name
if flag_norm == 0:
    if flag_itype == 0:
        outputname1 = "im_"+inputname+"_maxTh_"+str(int(max_th))+".jpg"
        outputname2 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+".jpg"
        outputname3 = "im_"+inputname+"_maxTh_"+str(int(max_th))+"_b.jpg"
    if flag_itype == 1:
        outputname1 = "im_"+inputname+"_maxTh_"+str(int(max_th))+".png"
        outputname2 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+".png"
        outputname3 = "im_"+inputname+"_maxTh_"+str(int(max_th))+"_b.png"
else:
    if flag_itype == 0:
        outputname1 = "im_"+inputname+"_maxTh_"+str(int(max_th))+"_n.jpg"
        outputname2 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+"_n.jpg"
        outputname3 = "im_"+inputname+"_maxTh_"+str(int(max_th))+"_n_b.jpg"
    if flag_itype == 1:
        outputname1 = "im_"+inputname+"_maxTh_"+str(int(max_th))+"_n.png"
        outputname2 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+"_n.png"
        outputname3 = "im_"+inputname+"_maxTh_"+str(int(max_th))+"_n_b.png"

print("Output image:\t",outputname1)
print("Output edges:\t",outputname2)

## -----------------------------------------------------------
## Perform real Canny edge extraction on the "original" image
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Real Canny extraction ##\n")

## Pre-blurry before Canny
img_previous = img
for it in range(1,n_blur+1):
    if it == 1:
        img_in = img

    else:
        img_in = img_previous

    if flag_blur == 0:
        img_blur = img_in
    elif flag_blur == 1:
        img_blur = cv.blur(img_in, (blur_level,blur_level))
    elif flag_blur == 2:
        img_blur = cv.GaussianBlur(img_in, (blur_level,blur_level), 0)

    img_previous = img_blur

## Canny extract
detected_edges = cv.Canny(img_blur, max_th*ratio, max_th, blur_level)

## Define mask & apply mask
mask = detected_edges != 0

if flag_Wedg == 1:
    detected_edges[mask] = 255
    edge = detected_edges
else:
    edge = img * (mask[:,:].astype(img.dtype))

if flag_info >= 1:
    cv.imshow("Canny extract (original)", edge)
    cv.waitKey(1)

    cv.imshow("Blurry image", img_blur) 
    cv.waitKey(1)
    # cv.imwrite(outputname3, img_blur, img_quality)

## -----------------------------------------------------------
## Perform transformation
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Perform transformation ##\n")

## Scaling size
img_s, r_img_s = ResizeWithAspectRatio(img, width=w_target)
if flag_info >= 1:
    cv.imshow("Scaled image", img_s) 
    cv.waitKey(1)

(h_s, w_s) = img_s.shape[:2]

edge_s, r_edge_s = ResizeWithAspectRatio(edge, width=w_target)
if flag_info >= 1:
    cv.imshow("Scaled edge", edge_s) 
    cv.waitKey(1)

print("Scaled image (H/W) =",h_s,"x",w_s)
print("r_img_s:\t",format(r_img_s,".3f"),"(expect = r_ref_ab*r_resize_arb)")
print("r_edge_s:\t",format(r_edge_s,".3f"),"(expect = r_ref_ab*r_resize_arb)\n")

## Rotation
img_srM = ModifiedRot(img_s, -1*angle_deg)
edge_srM = ModifiedRot(edge_s, -1*angle_deg)

if flag_info >= 1:
    cv.imshow("Mod_rot Image", img_srM)
    cv.waitKey(1)
    
    cv.imshow("Mod_rot Edge", edge_srM)
    cv.waitKey(1)

# cv.imwrite(outputname1, img_srM, img_quality)
# cv.imwrite(outputname2, edge_srM, img_quality)
height, width = edge_srM.shape[:2]
# new_width = int(width * 0.5)
# new_height = int(height * 0.5)
# edge_srM2 = cv.resize(edge_srM, (new_width, new_height), interpolation = cv.INTER_LINEAR)
## -----------------------------------------------------------
## Overlay
## -----------------------------------------------------------
## Direct replace edge color on image
# img_srM_rgb_cmp = cv.cvtColor(img_srM,cv.COLOR_GRAY2RGB)

# ix, jx = np.where(edge_srM > 255/2)
# print("find where",img_srM_rgb_cmp[ix,jx])

# img_srM_rgb_cmp[ix,jx] = [(150,255,255)]

# if flag_info >= 0:
#     cv.imshow("img_temp2", img_srM_rgb_cmp)
#     cv.waitKey(1)
cv.destroyWindow("Image preview")

## -----------------------------------------------------------
## Auto measurements
## -----------------------------------------------------------
print("\n-----------------------------------------------------------")
print("## Auto measurement ##\n")
if flag_autoM == 1:

    ## Select image
    edge_srM_rgb = cv.cvtColor(edge_srM,cv.COLOR_GRAY2RGB)

    ## Define plugs to be extracted
    if flag_Qs == 1:
        try:
            n_extract = float(input("How many plugs to be extracted? "))
        except ValueError:
            print("Not a number")
    else:
        n_extract = n_plugs_fix
    print("Plugs to be extracted:",n_extract,"\n")

    i_loop = 1
    while i_loop <= n_extract:

        print("\n--------------------")
        print("Loop counts= ",i_loop,",")

        cv.imshow(auto_name, edge_srM_rgb) 

        ## User define a reference point
        coords = []
        input_param =[edge_srM_rgb,auto_name]
        cv.setMouseCallback(auto_name, drawCross, input_param)

        print("Please select a point close to the center bottom of the extraction domain.\n")
        cv.waitKey(0)

        ## Define box
        coords = np.reshape(coords,(1,2))
        P_bot_x = coords[0,0]
        P_bot_y = coords[0,1]

        H_box_pix = int(H_box_nm*pix_nm_ref)
        W_box_pix = int(W_box_nm*pix_nm_ref)

        n_slice = int(H_box_nm//delta_nm)

        x_slice_L = P_bot_x-(W_box_pix//2)
        x_slice_R = P_bot_x+(W_box_pix//2)  

        print("L_ref, nm_pix_ref =",L_ref," [nm],",nm_pix_ref," [nm/pixel]")
        print("Pix_ref, pix_nm_ref =",P_ref," [pixel],",pix_nm_ref," [pixel/nm]")
        print("W_box_nm, H_box_nm =",W_box_nm," [nm],",H_box_nm," [nm]")
        print("W_box_pix, H_box_pix =",W_box_pix," [pixels],",H_box_pix," [pixels]")
        print("n_slice =",n_slice,"\n")

        ## Text
        font = cv.FONT_HERSHEY_SIMPLEX

        ## Draw measure lines
        for it in range(1,n_slice+1):
            y_slice = P_bot_y-pix_nm_ref*it*delta_nm
            cv.line(edge_srM_rgb, (x_slice_L,y_slice), (x_slice_R,y_slice), (200, 230, 200), 1)
            if it%2 == 0:
                cv.putText(edge_srM_rgb,str(int(it)),(x_slice_L-30,y_slice+4), font, 0.3, (77, 208, 225), 1, cv.LINE_AA)
            else:
                cv.putText(edge_srM_rgb,str(int(it)),(x_slice_L-15,y_slice+4), font, 0.3, (77, 208, 225), 1, cv.LINE_AA)
                
        ## Draw box boundary
        cv.rectangle(edge_srM_rgb, (x_slice_L, P_bot_y), (x_slice_R, P_bot_y-H_box_pix), (255, 255, 0), 1)

        cv.imshow(auto_name, edge_srM_rgb)
        cv.waitKey(1)

        ## Define memory for measurement output
        ## format (n_islands,y_pix,x_via_L_pix,x_via_R_pix,x_seam_L_pix,x_seam_R_pix,y_nm,via_CD_nm,seam_CD_nm)
        extract_data = np.zeros((n_slice,9))
        extract_data_temp = np.zeros((n_slice,9))

        indx_data = np.zeros((n_slice,4))
        indx_data_temp = np.zeros((n_slice,4))

## Copy for preview
        edge_srM_rgb_temp = edge_srM_rgb.copy()
        edge_srM_rgb_backup = edge_srM_rgb.copy()
        cv.imshow("Auto measu. preview", edge_srM_rgb_temp)
        cv.waitKey(1)

        for it in range(1,n_slice+1):
            y_slice = P_bot_y-pix_nm_ref*it*delta_nm
            ycut_data = edge_srM[y_slice,x_slice_L:x_slice_R]
##            print("x_slice_L,x_slice_R,y_slice =",x_slice_L,",",x_slice_R,",",y_slice)
##            print("ycut_data =",ycut_data)

            ycut_out = myislandinfo(ycut_data, trigger_val=0)[0]
##            print("ycut_out =",ycut_out)
##            print("length ycut_out =",len(ycut_out))

## skip if data not found
            if len(ycut_out) > 0:
                x_start, x_end = list(zip(*ycut_out))
                x_start = np.array(x_start)
                x_end = np.array(x_end)
            else:
                extract_data_temp[it-1,0] = 0
                extract_data_temp[it-1,1] = y_slice
                extract_data_temp[it-1,2] = 0
                extract_data_temp[it-1,3] = 0
                extract_data_temp[it-1,4] = 0
                extract_data_temp[it-1,5] = 0
                extract_data_temp[it-1,6] = 0
                extract_data_temp[it-1,7] = 0
                extract_data_temp[it-1,8] = 0
                continue
##            print("x_start =",x_start)
##            print("x_end =",x_end)

            x_start_pix = x_start+x_slice_L
            x_end_pix = x_end+x_slice_L
            x_center_pix = (x_start_pix+x_end_pix)//2
            x_center_pix_s = x_center_pix[1:-1]

##            print("x_start_pix=",x_start_pix)
##            print("x_end_pix=",x_end_pix)
##            print("x_center_pix=",x_center_pix)
##            print("x_center_pix_s=",x_center_pix_s)

            if len(x_start) >= 1:
                x_pos1_L = x_center_pix[0]
                cv.circle(edge_srM_rgb_temp, (x_pos1_L, y_slice), 2, (0, 150, 255), 1)
                indx_data_temp[it-1,0] = 1
            else:
                x_pos1_L = 0

##            print("x_pos1_L=",x_pos1_L)
            
            if len(x_start) >= 2:
                x_pos1_R = x_center_pix[-1]
                x_pos1_M = (x_pos1_L+x_pos1_R)//2
                cv.circle(edge_srM_rgb_temp, (x_pos1_R, y_slice), 2, (0, 150, 255), 1)
                via_CD_pix = (x_pos1_R-x_pos1_L)+1
                indx_data_temp[it-1,3] = len(x_start)

##                print("x_pos1_M=",x_pos1_M)
            else:
                x_pos1_R = 0
                via_CD_pix = 0

##            print("x_pos1_R=",x_pos1_R)
            
            if len(x_start) >= 3:
                array_L = x_center_pix_s[np.where(x_center_pix_s <= x_pos1_M)]
                array_R = x_center_pix_s[np.where(x_center_pix_s > x_pos1_M)]

##                print("array_L=",array_L)
##                print("array_R=",array_R)
                
                if len(array_L) >= 1 and len(array_R) >= 1:
                    x_pos2_L = array_L[-1]
                    x_pos2_R = array_R[0]
                    cv.circle(edge_srM_rgb_temp, (x_pos2_L, y_slice), 2, (255, 102, 255), 1)
                    cv.circle(edge_srM_rgb_temp, (x_pos2_R, y_slice), 2, (255, 102, 255), 1)
                    seam_CD_pix = (x_pos2_R-x_pos2_L)+1
                    indx_data_temp[it-1,1] = len(array_L)+1
                    indx_data_temp[it-1,2] = len(x_start)-len(array_R)

                if len(array_L) >= 1 and len(array_R) < 1:
                    if len(array_L) >= 2:
                        x_pos2_L = array_L[-2]
                        x_pos2_R = array_L[-1]
                        cv.circle(edge_srM_rgb_temp, (x_pos2_L, y_slice), 2, (255, 102, 255), 1)
                        cv.circle(edge_srM_rgb_temp, (x_pos2_R, y_slice), 2, (255, 102, 255), 1)
                        seam_CD_pix = (x_pos2_R-x_pos2_L)+1
                        indx_data_temp[it-1,1] = len(array_L)
                        indx_data_temp[it-1,2] = len(array_L)+1
                    else:
                        x_pos2_L = array_L[-1]
                        x_pos2_R = 0
                        cv.circle(edge_srM_rgb_temp, (x_pos2_L, y_slice), 2, (255, 102, 255), 1)
                        seam_CD_pix = 0
                        indx_data_temp[it-1,1] = len(array_L)+1
            
                if len(array_L) < 1 and len(array_R) >= 1:
                    if len(array_R) >= 2:
                        x_pos2_L = array_R[0]
                        x_pos2_R = array_R[1]
                        cv.circle(edge_srM_rgb_temp, (x_pos2_L, y_slice), 2, (255, 102, 255), 1)
                        cv.circle(edge_srM_rgb_temp, (x_pos2_R, y_slice), 2, (255, 102, 255), 1)
                        seam_CD_pix = (x_pos2_R-x_pos2_L)+1
                        indx_data_temp[it-1,1] = len(x_start)-len(array_R)
                        indx_data_temp[it-1,2] = len(x_start)-len(array_R)+1
                    else:
                        x_pos2_L = array_R[0]
                        x_pos2_R = 0
                        cv.circle(edge_srM_rgb_temp, (x_pos2_L, y_slice), 2, (255, 102, 255), 1)
                        seam_CD_pix = 0
                        indx_data_temp[it-1,1] = len(x_start)-len(array_R)
            else:
                x_pos2_L = 0
                x_pos2_R = 0
                seam_CD_pix = 0

##            print("x_pos2_L=",x_pos2_L)
##            print("x_pos2_R=",x_pos2_R)
##            print("pos_check=",x_pos1_L,x_pos2_L,x_pos2_R,x_pos1_R)

            h_nm = (P_bot_y-y_slice)/pix_nm_ref
            via_CD_nm = via_CD_pix/pix_nm_ref
            seam_CD_nm = seam_CD_pix/pix_nm_ref

            extract_data_temp[it-1,0] = len(x_start_pix)
            extract_data_temp[it-1,1] = y_slice
            extract_data_temp[it-1,2] = x_pos1_L
            extract_data_temp[it-1,3] = x_pos1_R
            extract_data_temp[it-1,4] = x_pos2_L
            extract_data_temp[it-1,5] = x_pos2_R
            extract_data_temp[it-1,6] = h_nm
            extract_data_temp[it-1,7] = via_CD_nm
            extract_data_temp[it-1,8] = seam_CD_nm

            cv.imshow("Auto measu. preview", edge_srM_rgb_temp)
            cv.waitKey(1)

        print("(n_islands,y_pix,via_L_pix,via_R_pix,seam_L_pix,seam_R_pix,y_nm,via_CD_nm,seam_CD_nm)")
        print("extract_data_temp:\n",extract_data_temp,"\n")
##        print("indx_data_temp:\n",indx_data_temp,"\n")

## Assume all correct
        add_corr = 1
        while add_corr != 0:

            ## User feedback lines to be corrected
            line_corr = [int(x) for x in input("Lines to be corrected (eg. 1 2 7), enter 0 if no: ").split()]
            print("lines to be corrected:",line_corr,"\n")

            if line_corr[0] == 0:
                add_corr = 0 
                break

            ## Correction section 
            line_indx = 0
            for it in range(1,n_slice+1):

                ## Define slice
                y_slice = P_bot_y-pix_nm_ref*it*delta_nm
                ycut_data = edge_srM[y_slice,x_slice_L:x_slice_R]
                h_nm = (P_bot_y-y_slice)/pix_nm_ref

                ## Find islands
                ycut_out = myislandinfo(ycut_data, trigger_val=0)[0]

                ## Skip if islands not found
                if len(ycut_out) > 0:
                    x_start, x_end = list(zip(*ycut_out))

                x_start = np.array(x_start)
                x_end = np.array(x_end)

                x_start_pix = x_start+x_slice_L
                x_end_pix = x_end+x_slice_L
                
                ## Perform corrections
                if it == line_corr[line_indx]:
                    print("Correct line no.",it," (find ",len(x_start)," islands)")

                    no1_old = int(indx_data_temp[it-1,0])
                    no2_old = int(indx_data_temp[it-1,1])
                    no3_old = int(indx_data_temp[it-1,2])
                    no4_old = int(indx_data_temp[it-1,3])
                    print("Old 4 islands (via_L seam_L seam_R via_R):",no1_old,no2_old,no3_old,no4_old)

                    while 0 < 1:
                        new_island_list = list(map(int, input("New 4 islands (via_L seam_L seam_R via_R)? ").split()))
                        if len(new_island_list) == 4:
                            no1 = new_island_list[0]
                            no2 = new_island_list[1]
                            no3 = new_island_list[2]
                            no4 = new_island_list[3]
                            indx_data_temp[it-1,0] = no1
                            indx_data_temp[it-1,1] = no2
                            indx_data_temp[it-1,2] = no3
                            indx_data_temp[it-1,3] = no4
                            print("Selected =",no1,",",no2,",",no3,",",no4,"\n")
                            break
                             
                    extract_data_temp[it-1,0] = np.count_nonzero(new_island_list)

                    if no1 == 0 and no2 == 0 and no3 == 0 and no4 == 0:
                        x_pos1_L = 0
                        x_pos1_R = 0
                        x_pos2_L = 0
                        x_pos2_R = 0
                        via_CD_pix = 0
                        seam_CD_pix = 0

                        via_CD_nm = via_CD_pix/pix_nm_ref
                        seam_CD_nm = seam_CD_pix/pix_nm_ref

                        extract_data_temp[it-1,1] = y_slice
                        extract_data_temp[it-1,2] = x_pos1_L
                        extract_data_temp[it-1,3] = x_pos1_R
                        extract_data_temp[it-1,4] = x_pos2_L
                        extract_data_temp[it-1,5] = x_pos2_R
                        extract_data_temp[it-1,6] = h_nm
                        extract_data_temp[it-1,7] = via_CD_nm
                        extract_data_temp[it-1,8] = seam_CD_nm
    
                    else:
                        if no1 > 0:
                            x_pos1_L = (x_start_pix[no1-1]+x_end_pix[no1-1])//2
                            extract_data_temp[it-1,2] = x_pos1_L
                        else:
                            x_pos1_L = 0
                            extract_data_temp[it-1,2] = x_pos1_L
                
                        if no4 > 0:
                            x_pos1_R = (x_start_pix[no4-1]+x_end_pix[no4-1])//2
                            extract_data_temp[it-1,3] = x_pos1_R
                        else:
                            x_pos1_R = 0
                            extract_data_temp[it-1,3] = x_pos1_R

                        if no1 > 0 and no4 > 0:
                            via_CD_pix = (x_pos1_R-x_pos1_L)+1
                            via_CD_nm = via_CD_pix/pix_nm_ref
                            extract_data_temp[it-1,7] = via_CD_nm
                        else:
                            via_CD_nm = 0
                            extract_data_temp[it-1,7] = via_CD_nm

                        if no2 > 0:
                            x_pos2_L = (x_start_pix[no2-1]+x_end_pix[no2-1])//2
                            extract_data_temp[it-1,4] = x_pos2_L
                        else:
                            x_pos2_L = 0
                            extract_data_temp[it-1,4] = x_pos2_L

                        if no3 > 0:
                            x_pos2_R = (x_start_pix[no3-1]+x_end_pix[no3-1])//2
                            extract_data_temp[it-1,5] = x_pos2_R
                        else:
                            x_pos2_R = 0
                            extract_data_temp[it-1,5] = x_pos2_R

                        if no2 > 0 and no3 > 0:
                            seam_CD_pix = (x_pos2_R-x_pos2_L)+1
                            seam_CD_nm = seam_CD_pix/pix_nm_ref
                            extract_data_temp[it-1,8] = seam_CD_nm
                        else:
                            seam_CD_nm = 0
                            extract_data_temp[it-1,8] = seam_CD_nm

                    if line_indx != len(line_corr)-1:
                        line_indx = line_indx+1

            ## Clear preview
            edge_srM_rgb_temp = edge_srM_rgb_backup.copy()
            cv.destroyWindow("Auto measu. preview")
            cv.imshow("Auto measu. preview", edge_srM_rgb_temp)

            ## Update preview                      
            for ix in range(1,n_slice+1):
                para0 = int(extract_data_temp[ix-1,0])
                para1 = int(extract_data_temp[ix-1,1])
                para2 = int(extract_data_temp[ix-1,2])
                para3 = int(extract_data_temp[ix-1,3])
                para4 = int(extract_data_temp[ix-1,4])
                para5 = int(extract_data_temp[ix-1,5])
                para6 = extract_data_temp[ix-1,6]
                para7 = extract_data_temp[ix-1,7]
                para8 = extract_data_temp[ix-1,8]

                if para0 >= 1:
                    cv.circle(edge_srM_rgb_temp, (para2, para1), 2, (0, 150, 255), 1)

                if para0 >= 2:
                    cv.circle(edge_srM_rgb_temp, (para3, para1), 2, (0, 150, 255), 1)

                if para0 >= 3:
                    cv.circle(edge_srM_rgb_temp, (para4, para1), 2, (255, 102, 255), 1)
    
                if para0 >= 4:
                    cv.circle(edge_srM_rgb_temp, (para5, para1), 2, (255, 102, 255), 1)

                cv.imshow("Auto measu. preview", edge_srM_rgb_temp)
                cv.waitKey(1)

            print("(n_islands,y_pix,via_L_pix,via_R_pix,seam_L_pix,seam_R_pix,y_nm,via_CD_nm,seam_CD_nm)")
            print("extract_data_temp:\n",extract_data_temp,"\n")
##            print("indx_data_temp:\n",indx_data_temp,"\n")

        ## Final data
        extract_data = extract_data_temp
        indx_data = indx_data_temp
        edge_srM_rgb = edge_srM_rgb_temp.copy()

        input_param =[edge_srM_rgb,auto_name]
        cv.setMouseCallback(auto_name, drawCross, input_param)
        cv.destroyWindow(auto_name)
        cv.imshow(auto_name, edge_srM_rgb)    
        cv.waitKey(1)

        print("(n_islands,y_pix,via_L_pix,via_R_pix,seam_L_pix,seam_R_pix,y_nm,via_CD_nm,seam_CD_nm)")
        print("extract_data:\n",extract_data,"\n")
##        print("indx_data:\n",indx_data,"\n")

        ## Output file
        if flag_norm == 0:
            outputname3 = "extract_"+inputname+"_maxTh_"+str(int(max_th))+"_L"+str(i_loop)+".xlsx"
        else:
            outputname3 = "extract_"+inputname+"_maxTh_"+str(int(max_th))+"_L"+str(i_loop)+"_n.xlsx"
        # np.savetxt(outputname3,extract_data,delimiter=' ',fmt='%.3e')

        wb_SW = openpyxl.Workbook()
        sh1 = wb_SW.create_sheet("THK",0)
        title_excel = ("n_islands","y_pix","via_L_pix","via_R_pix","seam_L_pix","seam_R_pix","y_nm","via_CD_nm","seam_CD_nm")
        sh1.append(title_excel)
        for ix in range(0,n_slice):
            data_temp = extract_data[ix]
            data_list = data_temp.tolist()
            sh1.append(data_list)        

        wb_SW.save(outputname3)
        ## Loop counts
        i_loop += 1

        if i_loop > n_extract:
            if flag_norm == 0:
                if flag_itype == 0:
                    outputname4 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+"_ext.jpg"        
                if flag_itype == 1:
                    outputname4 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+"_ext.png"
            else:
                if flag_itype == 0:
                    outputname4 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+"_ext_n.jpg"
                if flag_itype == 1:
                    outputname4 = "ed_"+inputname+"_maxTh_"+str(int(max_th))+"_ext_n.png"
            # cv.imwrite(outputname4, edge_srM_rgb, img_quality)
            input("Press Enter to exit...\n")       

## Exit the code
cv.destroyAllWindows()
sys.exit()
