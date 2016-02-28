import cv2
import numpy as np
import util as ut

cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
i=1
j=1
name=""
while(cap.isOpened()):
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th1 = cv2.threshold(gray.copy(),20,255,cv2.THRESH_BINARY)
	_,contours,hierarchy = cv2.findContours(th1.copy(),cv2.RETR_EXTERNAL, 2)
	cnt=ut.getMaxContour(contours,4000)
	if cnt!=None:
		x,y,w,h = cv2.boundingRect(cnt)
		imgT=img[y:y+h,x:x+w]
		imgT=cv2.bitwise_and(imgT,imgT,mask=th1[y:y+h,x:x+w])
		imgT=cv2.resize(imgT,(200,200))
		cv2.imshow('Trainer',imgT)
	cv2.imshow('Frame',img)
	cv2.imshow('Thresh',th1)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	if k == ord('s'):
		name=str(i)+"_"+str(j)+".jpg"
		cv2.imwrite('TrainData/'+name,imgT)
		if(j<20):
			j+=1
		else:
			while(0xFF & cv2.waitKey(0)!=ord('n')):
				j=1
			j=1
			i+=1
		

cap.release()        
cv2.destroyAllWindows()
