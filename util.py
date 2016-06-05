import cv2 
import numpy as np
import svm_train as st
import subprocess
import math

#get the screen res
def get_screen_res():
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return {'w': resolution[0], 'h': resolution[1]}


#Get the biggest Controur
def getMaxContour(contours,minArea=200):
    maxC=None
    maxArea=minArea
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>maxArea):
            maxArea=area
            maxC=cnt
    return maxC

def getContourBiggerThan(contours,minArea=200,maxArea=9000):
    maxC=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<maxArea and area>minArea:
            maxC.append(cnt)
    return maxC


    
#Get Gesture Image by prediction
def getGestureImg(cnt,img,th1,model):
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    imgT=img[y:y+h,x:x+w]
    imgT=cv2.bitwise_and(imgT,imgT,mask=th1[y:y+h,x:x+w])
    imgT=cv2.resize(imgT,(200,200))
    imgTG=cv2.cvtColor(imgT,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgTG',imgTG)
    resp=st.predict(model,imgTG)
    img=cv2.imread('TrainData/'+str(int(resp[0])+1)+'_2.jpg')
    return img,str(int(resp[0])+1)

#get slope of line 
def getSlope(pt1,pt2):
    x1,y1=pt1[0],pt1[1]
    x2,y2=pt2[0],pt2[1]
    m=float((y2-y1))/float((x2-x1))
    return math.degrees(math.atan(m))

def getDist(p1,p2):
    x1,y1=p1[0],p1[1]
    x2,y2=p2[0],p2[1]
    d=((x1-x2)**2+(y1-y2)**2)**.5
    return int(d)