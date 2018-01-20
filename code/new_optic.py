import cv2  
import numpy as np  
from PIL import Image  
import cv2.cv as cv  
import os
import sys
import argparse as ap

"""
author: ISS-Kerui
date:2018-01-20
This script USES the optical flow  to process dataset images.
"""


parser = ap.ArgumentParser()
parser.add_argument('-d', "--dataset_path", help="Path to dataset(including train data and test data)",
        required=True)
args = vars(parser.parse_args())

path = args["dataset_path"]
os.mkdir(path+'_optic')
for dataset in os.listdir(path):
    if os.path.exists(path+'_optic/'+dataset) == False:
        os.mkdir(path+'_optic/'+dataset)
        path2 = path+'/'+dataset
        path2_op = path+'_optic/'+dataset
    for clas_name in os.listdir(path2):
        if os.path.exists(path2_op+'/'+'clas_name') == False:
            os.mkdir(path2_op+'/'+clas_name)
        path3 = path2+'/'+clas_name
        path3_op = path2_op+'/'+clas_name
        for v in os.listdir(path3):
            new_v = os.path.join(path3,v)
            os.mkdir(path3_op+'/'+v)
            start_img = []
            later_img =[]
            num = 0
            for pic in os.listdir(new_v):
                if pic.split('.')[-1] == 'jpg':
                    new_pic = os.path.join(new_v,pic)
                    
                    
                    
                    if num ==0:
                        start_img = cv2.imread(new_pic)  
                        num +=1
                       
                        

                    
                    later_img = cv2.imread(new_pic)  
                    #cv2.imwrite(path+'_optic/'+v+"/frame_"+str(num)+'.jpg', mask)  

                    start_img=np.array(start_img)  

                    later_img=np.array(later_img)  

                    prvs = cv2.cvtColor(start_img,cv2.COLOR_BGR2GRAY)  
                    next = cv2.cvtColor(later_img,cv2.COLOR_BGR2GRAY)  

                      
                      
                      
                      
                    flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5,3,4,5,7,1.2, 1)  
                      
                      
                    hsv = np.zeros_like(start_img)  
                    hsv[...,1] =255 
                    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])  
                    hsv[...,0] = ang*180/np.pi/2  
                    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)  
                    RGB= cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)  
                    GRAY = cv2.cvtColor(RGB,cv2.COLOR_BGR2GRAY)  
                    # img = cv2.add(RGB,start_img)
                    rows,cols,channels = start_img.shape  
                    roi = start_img[0:rows, 0:cols ]  
                    ret, mask = cv2.threshold(GRAY, 10, 255, cv2.THRESH_BINARY)  
                    mask_inv = cv2.bitwise_not(mask)  
          
                    # Now black-out the area of logo in ROI  
                    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
                      
                    # Take only region of logo from logo image.  
                    img2_fg = cv2.bitwise_and(RGB,RGB,mask = mask)  
                      
                    # Put logo in ROI and modify the main image 
                    print img1_bg.shape 
                    print img2_fg.shape 
                    dst = cv2.add(img1_bg,img2_fg)  
                    start_img[0:rows, 0:cols ] = dst  
                  
                    cv2.imwrite(path3_op+'/'+v+"/frame_"+str(num)+'.jpg', start_img) 

                    start_img = later_img.copy()
                    num += 1
          

      
  

 
  
