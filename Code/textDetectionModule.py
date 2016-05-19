import cv2
import numpy as np
import sys
import scipy.misc
import operator 
from operator import itemgetter

#rectlist2 has big rectangles
#now corresponding to each rectangle there, assign small rectangles to it
class WordRectangle:
    charCount=0
    regions = [] 
    hulls = []
    
    def __init__(self,x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h

    characterRectangle=[]

    def addCharRect(self,x):
        self.characterRectangle.append(x)
        self.charCount+=1

#font for cv text
font = cv2.FONT_HERSHEY_SIMPLEX

def getText(img,arg2,arg3):
	y = np.shape(img)[0]
	x = np.shape(img)[1]

	#negate document
	img = cv2.resize(img,(0,0),fx=1,fy=1)
	img = 255-img

	#create copy of image
	vis2 = img.copy()

	#dilate image
	kernel = np.ones((20,0),np.uint(8))
	vis2 = cv2.dilate(vis2,kernel,iterations=21)
	cv2.imshow('Dilated Image.',vis2)
	#scipy.misc.imsave('imgDilated'+'.png',vis2)
	cv2.waitKey(0)

	#find mser regions for big rectangles in dilated image
	#10k is min area
	#max area = 0.5 of area of image
	#min area = 0.01 area of image
	mser = cv2.MSER_create(1, int(0.01*y*x), int(0.5 * y*x), 0.05, 0.02, 200, 1.01, 0.003, 5)
	regions = mser.detectRegions(vis2, None)

	#store the big rectangles detected in rectList2
	rectList2 = []
	for i in range(len(regions)):
	    x,y,w,h = cv2.boundingRect(regions[i])
	    if(w*h>int(arg3)):
	        rectList2.append([x,y,w,h])

	#group the rectangles to pair similar rectangle
	rectList2,weights = cv2.groupRectangles(rectList2,int(arg2))#,float(arg3))

	WordRectangleArray = []

	#create copy of original image
	vis = img.copy()
	vis = cv2.cvtColor(vis,cv2.COLOR_GRAY2BGR)

	#draw the rectangles on vis
	for i in range(len(rectList2)):
	    x = rectList2[i][0]
	    y = rectList2[i][1]
	    w = rectList2[i][2]
	    h = rectList2[i][3]
	    cv2.rectangle(vis,(x,y),(x+w,y+h),(255,0,0),2)
	    WordRectangleArray.append(WordRectangle(x,y,w,h))

	#have big rectangles
	#find small rectangles
	for i in range(len(rectList2)):
	    mser = cv2.MSER_create(1, 0, 94400, 0.05, 0.02, 200, 1.01, 0.003, 5)
	    x0 = rectList2[i][0]
	    y0 = rectList2[i][1]
	    w0 = rectList2[i][2]
	    h0 = rectList2[i][3]

	    WordRectangleArray[i].regions = mser.detectRegions(img[y0:y0+h0,x0:x0+w0], None)
	    rectList = []

	    for j in range(len( WordRectangleArray[i].regions)):
	        x,y,w,h = cv2.boundingRect( WordRectangleArray[i].regions[j])

	        #0.9 added to remove the big rectangle
	        if(w*h>int(arg3) and w*h<0.9*w0*h0):
	           rectList.append([x0+x,y0+y,w,h])
	           rectList.append([x0+x,y0+y,w,h])
	         
	    rectList,weights = cv2.groupRectangles(rectList,int(arg2),0.1)
	    avg = 0 
	    sums = 0
	    for k in range(len(rectList)):
	    	sums+= rectList[k][2]*rectList[k][3]

	    if(len(rectList)>0):
	    	avg = sums/len(rectList)
	   	rectList = filter(lambda x: x[2]*x[3]>0.8*avg ,rectList)

	    rectList = sorted(rectList,key=itemgetter(0))
	    WordRectangleArray[i].characterRectangle = rectList

	def compare(x,y):
		if(x<0.9*y):
			return -1
		elif(x>0.9*y):
			return 1
		else:
			return 0
	#modified sort that allows for variance
	WordRectangleArray = sorted(WordRectangleArray,cmp=compare,key=(operator.attrgetter('y')))
	#no need to sort in x, already sorted
	count=0

	for i in range(len(WordRectangleArray)):
	    x0 = WordRectangleArray[i].x
	    y0 = WordRectangleArray[i].y
	    w0 = WordRectangleArray[i].w
	    h0 = WordRectangleArray[i].h

	    tmp = img[y0:y0+h0,x0:x0+w0]  

	    for j in range(len(WordRectangleArray[i].characterRectangle)):
	        count+=1
	        x = WordRectangleArray[i].characterRectangle[j][0]
	        y = WordRectangleArray[i].characterRectangle[j][1]
	        w = WordRectangleArray[i].characterRectangle[j][2]
	        h = WordRectangleArray[i].characterRectangle[j][3] 
	        cv2.rectangle(vis,(x,y),(x+w,y+h),(255,0,0),2)
	        cv2.putText(vis,str(count),(x,y),font, 1,(255,255,255),2,cv2.LINE_AA)

	    
	cv2.imshow('imgWithTextBoxes', vis)
	cv2.waitKey(0)
	#scipy.misc.imsave('imgWithTextBoxes'+'.png',vis)
	#cv2.destroyAllWindows()
	return WordRectangleArray
