import cv2 
import numpy as np
import svm_train as st

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

#Get the stop of Gesture
def gestureStop(lastPoint,cx,cy):
    if lastPoint[0]<=cx+10 and lastPoint[0]>=cx-10 and lastPoint[1]>=cy-10 and lastPoint[1]<=cy+10:
        return True
    else:
        return False

#Save Gesture to File
def saveToFile(gstr,gname):
    f=open('gesData.txt','r+')
    f.write(gname+'_')
    f.write(gstr+'#')
    f.close()

#Read Gesture From File
def loadGesDB():
    gobjs=[]
    f=open('gesData.txt','r')
    data=f.read()
    gDb=GestureDatabase()
    i=1
    print 'Training Gesture Model'
    if data!='':
        for record in data.split('#'):
            
            if record!='':
                gDb.add_gesture(gDb.str_to_gesture(record.split('_')[1]))
                gobjs.append((record.split('_')[0],gDb.str_to_gesture(record.split('_')[1])))
        return gobjs,gDb
    else:
        return gobjs,gDb

    
#Get Gesture Image by prediction
def getGestureImg(cnt,img,th1,model):
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    imgT=img[y:y+h,x:x+w]
    imgT=cv2.bitwise_and(imgT,imgT,mask=th1[y:y+h,x:x+w])
    imgT=cv2.resize(imgT,(200,200))
    imgTG=cv2.cvtColor(imgT,cv2.COLOR_BGR2GRAY)
    resp=st.predict(model,imgTG)
    img=cv2.imread('TrainData/'+str(int(resp[0])+1)+'_2.jpg')
    return img