import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
import sys
import os
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import itertools
import operator 
#user defined
import binarizeModule
import textDetectionModule
import trainingModule
import featuresModule

'''
first binarize the image
'''

imgBlurredThresholded = binarizeModule.binarize(sys.argv[1])

'''
now we identify the text regions in image using MSER
To identify word regions first we, dilate the image and 
then apply MSER.
Once we get the word regions, we split them into best fitting
rectangles and apply MSER inside it again to get the individual
characters. 
To sort the characters in order of human reading, we first sort 
the individual words in Y axis and then X axis using a stable sort.
Inside each word box, the characters are sorted in order of X axis.

The output of this, each character box is then passed to feature extraction and classification
algorithm. 
Features extracted are : 
1. Mean x coordinate of character in box
2. Mean y coordinate of character in box
3. Black Islands
4. A n*n grid representing the character box where the grid entry has value 1 
even if there is a single character pixel in that portion of grid else 0.
This is eventually flattened to a 1-d list.
5. Hu-Moments
'''


img = cv2.imread('imageBlurredThresholded.jpg',0)
WordRectangleArray = textDetectionModule.getText(img,1,50)
#print (len(WordRectangleArray))

img = cv2.resize(img,(0,0),fx=1,fy=1)
img = 255-img
count=1

'''
get the knn and svm classifiers
'''
classifiers = trainingModule.getClassifiers(sys.argv[2])
knn = classifiers[0]
svm = classifiers[1]

def binarizeStandardize(img):
	#img = cv2.imread('/home/shinchan/patternrecognition/char/Sample001/'+ip,0)
	x = np.shape(img)[1]
	y = np.shape(img)[0]
	#img = 255-img
	temp,otsuimg = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	white  = 0
	black = 0
	for i in range(x):
		for j in range(y):
			if(otsuimg[j][i]==0):
				black+=1
			else:
				white+=1
	#otsuimg = cv2.cvtColor(otsuimg,cv2.COLOR_GRAY2BGR)
	#print black,white
	if black<white:
		#print("black")
		otsuimg = 255-otsuimg #make all images white on black
	#else:
	#	print("white")
		#otsuimg = 255 -otsuimg
	return otsuimg

KNNWords = []
SVMWords = []

for i in range(len(WordRectangleArray)):
	count = 1
	KNNWord = ''
	SVMWord = ''
	for j in range(len(WordRectangleArray[i].characterRectangle)):
		x = WordRectangleArray[i].characterRectangle[j][0]
		y = WordRectangleArray[i].characterRectangle[j][1]
		w = WordRectangleArray[i].characterRectangle[j][2]
		h = WordRectangleArray[i].characterRectangle[j][3]
		charBox = img[y:y+h,x:x+w]
		charBox = binarizeStandardize(charBox)
		scipy.misc.imsave('sciWord'+str(i+1)+'-'+str(count)+'.png',charBox) 


		charFeatures = featuresModule.getFeatureVector('',charBox)
		n = len(charFeatures)
		#print n
		charFeatures = np.array(charFeatures)
		#print charFeatures
		charNames = charFeatures[0]
		charFeatures = charFeatures[1:n]
		#print 'sciWord'+str(i+1)+'-'+str(count)+'.png',str(knn.predict(charFeatures))
		#print charFeatures
		KNNWord = KNNWord+ str(knn.predict(charFeatures)[0])
		SVMWord = SVMWord + str(svm.predict(charFeatures)[0])
		count+=1
	
	KNNWords.append(KNNWord)
	SVMWords.append(SVMWord)
	
print knn
print svm
print "KNN", KNNWords
print "SVM", SVMWords
