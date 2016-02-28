import cv2
import numpy as np
import util as ut
import svm_train as st 
import time

model=st.trainSVM(9)
#create and train SVM model each time coz bug in opencv 3.1.0 svm.load() https://github.com/Itseez/opencv/issues/4969
cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX


while(cap.isOpened()):

	t=time.time()
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th1 = cv2.threshold(gray.copy(),20,255,cv2.THRESH_BINARY)
	_,contours,hierarchy = cv2.findContours(th1.copy(),cv2.RETR_EXTERNAL, 2)
	cnt=ut.getMaxContour(contours,4000)
	if cnt!=None:
		gesture=ut.getGestureImg(cnt,img,th1,model)
		cv2.imshow('PredictedGesture',gesture)
		
	fps=int(1/(time.time()-t))
	cv2.putText(img,"FPS: "+str(fps),(50,50), font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Frame',img)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	

cap.release()        
cv2.destroyAllWindows()
