#file to read all files in directory and get filename
#for files in directory then split the name of file depend on "-"
import numpy as np
import os
import glob
from os.path import basename
import cv2
from decimal import *
################# structure dataset 
#####################
print 'hello'
dir_name="/home/obaid/temp_test/*.*"
dir_dest="/home/obaid/dest_colour"
files=glob.glob(dir_name)       #files equal all list files in path
count=1
u=0
ccc=1
for file in files:              # file will contain path of file
    x=os.path.splitext(file)[0] # split the path to directory and filename without extension
    z=basename(x)               # z extract only name fal
    h=z.split("_")

    ###########read files
    vidcap = cv2.VideoCapture(file)
    success,image = vidcap.read()
    frm_no=int(vidcap.get (cv2.CAP_PROP_POS_FRAMES )) #the current frame
    length=int(vidcap.get (cv2.CAP_PROP_FRAME_COUNT )) #the number of frames
    a= length/8
    b=length%8
    print 'length',length
    print "a",a,"b",b #remeber of frame 
  
    success = True
    
    for frm in range(8):   # frm variable represent the stage 
        os.chdir (dir_dest)
        print "frm",frm,frm_no,"b",b
        
        cv2.imwrite("%d%s_%s_%s_%s.jpg" % (frm,h[1],h[0],h[2],frm_no), image)   
        if b<=0:
            c=0
        else:
            c=1
        
        aall=a+c-1
        print "aall",aall
        for i in range(aall):
            
            success,image = vidcap.read()
            frm_no=int(vidcap.get (cv2.CAP_PROP_POS_FRAMES )) #the current frame
            cv2.imwrite("%d%s_%s_%s_%s.jpg" % (frm,h[1],h[0],h[2],frm_no), image)
            print "i",i,frm_no,"c",c
        b=b-1
        success,image = vidcap.read()
        frm_no=int(vidcap.get (cv2.CAP_PROP_POS_FRAMES )) #the current frame
