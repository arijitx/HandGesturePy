"""
#Author : Arijit Mukherjee
#Date 	: June 2016
#B.P. Poddar Institute of Management and Technology
#Inteligent Human-Computer Interaction with depth prediction using normal webcam and IR leds
#Inspired by : http://research.microsoft.com/pubs/220845/depth4free_SIGGRAPH.pdf


Demo application to predict hand-pose from a set of test data 
"""

#Importing Opencv and Numpy
import cv2
import numpy as np

#Importing our dependencies
import util as ut
import svm_train as st 



import time


#create and train SVM model each time coz bug in opencv 3.1.0 svm.load() https://github.com/Itseez/opencv/issues/4969
model=st.trainSVM(9,20,'TrainData2')
move_text={'1':'GRAB','2':'Bless','3':'Rock','4':'Stop','5':'ThumbsUp','6':'Victory','7':'Stop2','8':'Left','9':'Right'}

#Camera and font initialization
cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX


#The main event loop
while(cap.isOpened()):
	move=''
	t=time.time()
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th1 = cv2.threshold(gray.copy(),150,255,cv2.THRESH_TOZERO)
	cv2.imshow('thresh',th1)
	_,contours,hierarchy = cv2.findContours(th1.copy(),cv2.RETR_EXTERNAL, 2)
	cnt=ut.getMaxContour(contours,4000)
	if cnt!=None:
		gesture,res=ut.getGestureImg(cnt,img,th1,model)
		cv2.imshow('PredictedGesture',cv2.imread('TrainData2/'+res+'_1.jpg'))
		move='         '+move_text[res]
		
	fps=int(1/(time.time()-t))
	cv2.putText(img,"FPS: "+str(fps)+move,(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Frame',img)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	

cap.release()        
cv2.destroyAllWindows()
