import os
import cv2
from sklearn.cross_validation import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import itertools



def xBar(ip):
	x = np.shape(ip)[1] #number of columns
	y = np.shape(ip)[0] #number of rows
	ret = 0
	#n =0 #if 0 can cause divison by 0 error
	n=1
	for i in range(x):
		for j in range(y):
			if(ip[j,i]>0):
				ret+=i
				n+=1
	ret = float(ret)/(n*x)
	return ret

def yBar(ip):
	x = np.shape(ip)[1] #number of columns
	y = np.shape(ip)[0] #number of rows
	ret = 0
	#n =0 
	n=1
	for i in range(x):
		for j in range(y):
			if(ip[j,i]>0):
				ret+=j
				n+=1
	ret = float(ret)/(n*y)
	return ret

#http://arxiv.org/pdf/0904.3650.pdf
def HuMoments(ip):
	temp = cv2.HuMoments(cv2.moments(ip.astype('uint8')))
	temp = temp.tolist()
	temp = list(itertools.chain.from_iterable(temp))
	return temp


def merge(img,i,j):
	h, w = img.shape[:2]
	if(i<0 or j<0 or i>w-1 or j>h-1):
		return img
	if(img[j][i] == 0):
		return img 
	img[j][i]=0
	img = merge(img,i-1,j)
	img = merge(img,i+1,j)
	img = merge(img,i,j-1)
	img = merge(img,i,j+1)
	return img

def findWhiteIslands(img):
	h, w = img.shape[:2]
	count = 0
	vis = img.copy()
	for i in range(w):
		for j in range(h):
			if(vis[j][i]>0):
				count += 1
				vis = merge(vis,i,j)
	return count

def findBlackIslands(img):
	h, w = img.shape[:2]
	count = 0
	vis = img.copy()
	for i in range(w):
		for j in range(h):
			if(vis[j][i]==0):
				count += 1
				vis = mergeBlack(vis,i,j)
	return count			

def mergeBlack(img,i,j):
	h, w = img.shape[:2]
	if(i<0 or j<0 or i>w-1 or j>h-1):
		return img
	if(img[j][i] > 0):
		return img 
	img[j][i]=1
	img = mergeBlack(img,i-1,j-1)
	img = mergeBlack(img,i-1,j)
	img = mergeBlack(img,i-1,j+1)
	img = mergeBlack(img,i+1,j)
	img = mergeBlack(img,i+1,j-1)
	img = mergeBlack(img,i+1,j+1)
	img = mergeBlack(img,i,j-1)
	img = mergeBlack(img,i,j+1)
	return img


def returnIslands(img):
	#assumes image is white text on black
	img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
	h, w = img.shape[:2]
	#dilate image
	kernel = np.ones((1,0),np.uint(8))
	img = cv2.dilate(img,kernel,iterations=1)
	_,img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
	return findBlackIslands(img)

def countCorners(img):
	img = cv2.GaussianBlur(img,(21,21),0)
	gray = img
	#add border
	gray = cv2.copyMakeBorder(gray,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
	
	# find Harris corners
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.001)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)
	# find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	cv2.imwrite('subpixel5.png',gray)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
	# Now draw them
	res = np.hstack((centroids,corners))
	res = np.int0(res)
	return len(res)


def isWhite(x):
	w = np.shape(x)[1]
	h = np.shape(x)[0]
	for i in range(w):
		for j in range(h):
			if(x[j][i]>0):
				return 1
	return 0

def grid(img,n):
	wBy3 = np.shape(img)[1]/n
	hBy3 = np.shape(img)[0]/n

	features = []

	for i in range(n):
		for j in range(n):
			x=img[hBy3*i:hBy3*(i+1),wBy3*j:wBy3*(j+1)]
			features.append( isWhite(x) )

	return features

def getFeatureVector(imageName,image):
	imageAsObject = image.astype(np.object,copy=False)
	name = imageName[0:imageName.find('.')]
	features = [name,xBar(imageAsObject),yBar(imageAsObject),returnIslands(image)]
	features.extend(grid(image,7))
	return features



