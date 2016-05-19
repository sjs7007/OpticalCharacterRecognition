import cv2
import sys
import scipy.misc

#binarizes image and saves it also
def binarize(imgname):
	#first binarize the image using adaptive thresholding 
	#image is taken as the 1st command line argument
	#imgraw = cv2.imread(sys.argv[1],0)
	imgraw = cv2.imread(imgname,0)

	cv2.imshow('Raw Image.',imgraw)
	cv2.waitKey(0)

	#blur to remove noise first 
	imgBlurred = cv2.GaussianBlur(imgraw,(5,5),0)
	imgBlurred = cv2.medianBlur(imgBlurred,5)

	imgBlurredThresholded = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	scipy.misc.imsave('imageBlurredThresholded.jpg',imgBlurredThresholded)
	cv2.imshow('Image after thresholding and noise removal.',imgBlurredThresholded)
	cv2.waitKey(0)
	return imgBlurredThresholded