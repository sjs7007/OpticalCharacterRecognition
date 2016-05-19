import cv2
import numpy as np
import sys
import scipy.misc
import os


def gridify(imgName,n):
	img = cv2.imread(imgName)
	w = np.shape(img)[1]
	h = np.shape(img)[0]
	wByn = w/n
	hByn = h/n

	for i in range(n):
		img[:,wByn*i]=(255,0,0)

	img[:,w-1]=(255,0,0)
	img[h-1,:]=(255,0,0)

	for j in range(n):
		img[hByn*j,:]=(255,0,0)

	x = os.path.basename(imgName)+'withGrid.jpg'
	print x

	scipy.misc.imsave(x,img)


gridify(sys.argv[1],int(sys.argv[2]))
