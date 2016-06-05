"""
#Author : Arijit Mukherjee
#Date 	: June 2016
#B.P. Poddar Institute of Management and Technology
#Inteligent Human-Computer Interaction with depth prediction using normal webcam and IR leds
#Inspired by : http://research.microsoft.com/pubs/220845/depth4free_SIGGRAPH.pdf




Demo Application to estimate hand-pose and triggering mouse events by hand gestures
and dynamic gesture recognition 8 DOF using moosegesture 
  up-left    up   up-right
         7   8   9

    left 4       6 right

         1   2   3
down-left   down  down-right
"""




#default python libraries
import time

#Opencv and dependencies
import cv2
import numpy as np

#our libraries
import util as ut
import svm_train as st 
import hand_util as hu

#other dependencies
from pymouse import PyMouse
import moosegesture as mges


#PyMouse the library to control mouse movements from python
m1 = PyMouse()


#capturing device 
cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)



#training the svm 
model=st.trainSVM(3,40,'TrainData')

#initilizing values
thresh=120
frame_count=0
color=(0,0,255)
res=ut.get_screen_res()
w_screen=int(res['w'])+200
h_screen=int(res['h'])+200
font = cv2.FONT_HERSHEY_SIMPLEX


#loop 1 to calculate the mean threshhold


while(cap.isOpened()):
	# for fps calc
	t=time.time()

	#capturing frame 
	_,img=cap.read()

	#converting frame to grayscale
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#seting up the roi for the hand postion
	cv2.rectangle(img,(270,165),(370,315),color,3)

	#fps calc
	fps=int(1/(time.time()-t))
	cv2.putText(img,"FPS: "+str(fps),(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Frame',img)
	frame_count+=1

	#getting input
	k = 0xFF & cv2.waitKey(10)
	if k==27:
		break

	if frame_count==80:
		color=(0,255,0)

	if frame_count==100:
		thresh=cv2.mean(gray[165:315,270:370])
		thresh=thresh[0]-15
		break

#initilizing values
pressed=False
trail=[]
mouse_enable=True
gesFound=0


#the main event loop
while(cap.isOpened()):
	t=time.time()

	#grabbing a frame
	_,img=cap.read()

	#preprocessing
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th1 =cv2.threshold(gray,thresh,255,cv2.THRESH_TOZERO)
	cv2.imshow('threshold',th1)
	
	# contour detection and getting the contours with minArea and maxArea
	_,contours,hierarchy = cv2.findContours(th1.copy(),cv2.RETR_EXTERNAL, 2)
	cnts=ut.getContourBiggerThan(contours,minArea=3000,maxArea=40000)

	#if no object in the frame check the trail list for any previous trail
	if len(cnts)==0:
		#get the gesture
		strokes = mges.getGesture(trail)
		trail=[]
		index=0
		maxLen=0
		for i in xrange(0,len(strokes)):
			if strokes[i]>maxLen:
				maxLen=strokes[i]
				index=i

		#find the gesture with the maximum consistent stroke length
		if maxLen>0:
			strokeText = mges.getGestureStr(strokes)
			strokeText=strokeText.split(' ')[index]
			gesFound=10



	#if gesture found show the gesture text for the next 10 frames
	if gesFound>0:
		cv2.putText(img,strokeText,(300,200), font,6,(255,255,255),10,cv2.LINE_AA)
		gesFound-=1



	#processing the contrours
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)

		#predicting the hand pose 
		_,resp=ut.getGestureImg(cnt,img,th1,model)

		#calculating the centroid of the hand
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])

		#mark the centroid in the image
		cv2.circle(img,(cx,cy),5,[0,255,0],-1)

		#get mouse location
		mx=int((int(w_screen)/640)*cx)
		my=int((int(h_screen)/480)*cy)
		print mx,my


		#always keep maximum of 20 points in the trail
		if len(trail)==20:
			trail.pop(0)

		#append the centroid to the trail
		trail.append((cx,cy))

		#mouse events by hand pose 
		if int(resp)==1 and mouse_enable:
			if pressed:
				pressed=False
				m1.release(mx,my)
			m1.move(mx,my)
		if int(resp)==2 and mouse_enable:
			pressed=True
			m1.press(mx,my)


		#put the hand pose on the display
		cv2.putText(img,resp,(x,y), font,1,(255,255,255),2,cv2.LINE_AA)
	

	#draw the trail
	for i in xrange(0,len(trail)-1):
		cv2.line(img,trail[i],trail[i+1],[0,0,255],2)


	#fps calc
	fps=int(1/(time.time()-t))
	cv2.putText(img,"FPS: "+str(fps),(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Frame',img)


	#key press events

	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	if k == ord('r'):
		mouse_enable=not mouse_enable
	


# release the capture resource and destroy all windows in the end 
cap.release()        
cv2.destroyAllWindows()
