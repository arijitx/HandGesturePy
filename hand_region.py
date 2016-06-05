#Importing Opencv and Numpy
import cv2
import numpy as np

#Importing our dependencies
import util as ut
import svm_train as st 
import hand_util as hu

import time


#create and train SVM model each time coz bug in opencv 3.1.0 svm.load() https://github.com/Itseez/opencv/issues/4969
model=st.trainSVM(9,20,'TrainData2')

kernel = np.ones((5,5),np.uint8)
#Camera and font initialization
cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX


#The main event loop
while(cap.isOpened()):
	t=time.time()
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th1 = cv2.threshold(gray.copy(),130,255,cv2.THRESH_BINARY)
	th1= cv2.erode(th1,kernel,iterations =2)
	cv2.imshow('thresh',th1)
	_,contours,hierarchy = cv2.findContours(th1.copy(),cv2.RETR_EXTERNAL, 2)
	cnt=ut.getMaxContour(contours,4000)
	

	if cnt!=None:
		gesture,res=ut.getGestureImg(cnt,img,th1,model)
		cv2.imshow('PredictedGesture',cv2.imread('TrainData2/'+res+'_1.jpg'))
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		cv2.circle(img,(cx,cy),5,[0,0,255],-1)
		#cv2.putText(img,res,(x,y), font,1,(255,255,255),2,cv2.LINE_AA)
		hull = cv2.convexHull(cnt,returnPoints = False)
		defects = cv2.convexityDefects(cnt,hull)
		if defects!=None:
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				if hu.angle_rad(np.subtract(start, far), np.subtract(end, far)) < hu.deg2rad(80):
					cv2.circle(img, far, 5, [0,0,255], -1)
				cv2.line(img,start,end,[0,255,0],2)
				#cv2.circle(img,far,5,[0,0,255],-1)
		
	

	fps=int(1/(time.time()-t))
	cv2.putText(img,"FPS: "+str(fps),(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Frame',img)
	

	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	

cap.release()        
cv2.destroyAllWindows()



