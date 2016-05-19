import os
import cv2
from sklearn.cross_validation import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import itertools

#user created
import featuresModule

def binarizeStandardize(img):
	x = np.shape(img)[1]
	y = np.shape(img)[0]
	temp,otsuimg = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	white  = 0
	black = 0
	for i in range(x):
		for j in range(y):
			if(otsuimg[j][i]==0):
				black+=1
			else:
				white+=1

	if black<white:
		otsuimg = 255-otsuimg #make all images white on black
	#else:
	return otsuimg

def getClassifiers(path):			
	features = []

	for imageName in os.listdir(path):
		tempImage = binarizeStandardize(cv2.imread(path+imageName,0))
		features.append(featuresModule.getFeatureVector(imageName,tempImage))

	n = np.shape(features)[1]

	features = np.array(features)
	x = features[:,1:n]
	y = features[:,0]

	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(x,y)

	svmClassifier = svm.SVC()
	svmClassifier.fit(x,y)

	return [knn,svmClassifier]

