"""
#Author : Arijit Mukherjee
#Date 	: June 2016
#B.P. Poddar Institute of Management and Technology
#Inteligent Human-Computer Interaction with depth prediction using normal webcam and IR leds
#Inspired by : http://research.microsoft.com/pubs/220845/depth4free_SIGGRAPH.pdf


Demo Application to play a game with just using ur hands 
"""


import cv2
import numpy as np
import util as ut
import svm_train as st 
import time
import hand_util as hu



cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX
model=st.trainSVM(3,40,'TrainData')
font = cv2.FONT_HERSHEY_SIMPLEX
thresh=120
frame_count=0
color=(0,0,255)


while(cap.isOpened()):

	t=time.time()
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.rectangle(img,(270,165),(370,315),color,3)
	fps=int(1/(time.time()-t))
	cv2.putText(img,"FPS: "+str(fps),(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Frame',img)
	k = 0xFF & cv2.waitKey(10)
	frame_count+=1
	if frame_count==80:
		color=(0,255,0)
	if frame_count==100:
		thresh=cv2.mean(gray[165:315,270:370])
		thresh=thresh[0]-15
		break


while(cap.isOpened()):
	t=time.time()
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th1 =cv2.threshold(gray,thresh,255,cv2.THRESH_TOZERO)
	cv2.imshow('threshold',th1)
	_,contours,hierarchy = cv2.findContours(th1.copy(),cv2.RETR_EXTERNAL, 2)
	cnts=ut.getContourBiggerThan(contours,minArea=3000,maxArea=40000)
	line=[]
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)
		_,resp=ut.getGestureImg(cnt,img,th1,model)
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		line.append((cx,cy))
		cv2.circle(img,(cx,cy),5,[0,255,0],-1)
		cv2.putText(img,resp,(x,y), font,1,(255,255,255),2,cv2.LINE_AA)
	if len(line)==2:
		pt1=line[0]
		pt2=line[1]
		ang=int(ut.getSlope(pt1,pt2))
		cv2.putText(img,'Angle-> '+str(ang),(400,50), font,1,(255,255,255),2,cv2.LINE_AA)
		if ang>0:
			cv2.putText(img,'RIGHT',((pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2), font,3,(255,255,255),2,cv2.LINE_AA)
		else:
			cv2.putText(img,'LEFT',((pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2), font,3,(255,255,255),2,cv2.LINE_AA)
		cv2.line(img,line[0],line[1],[0,0,255],5)
	fps=int(1/(time.time()-t))
	cv2.putText(img,"FPS: "+str(fps),(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Frame',img)
	old_gray = gray.copy()
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break

cap.release()        
cv2.destroyAllWindows()
