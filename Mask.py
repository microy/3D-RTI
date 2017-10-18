#! /usr/bin/env python
# -*- coding:utf-8 -*-

# External dependencies
import glob
import sys
import cv2
import numpy as np

# Create a mask for the images
def CreateMask( images ) :
	temp = np.mean( images, axis=0 ).astype( np.uint8 )
#	temp = cv2.resize( temp, None, fx=0.3, fy=0.3 )
#	temp[ temp > 50 ] = 255
# noise removal
#	ret, thresh = cv2.threshold( temp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
	kernel = np.ones( (3,3), np.uint8 )
	opening = cv2.morphologyEx( temp, cv2.MORPH_OPEN, kernel )
	sure_bg = cv2.dilate( opening, kernel, iterations=3 )
	# Otsu's thresholding after Gaussian filtering
	# blur = cv2.GaussianBlur(temp,(5,5),0)
	# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# im2, contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(im2, contours, -1, (0,255,0), 3)
	# cv2.imshow( 'temp', cv2.resize( im2.astype( np.uint8 ), None, fx=0.3, fy=0.3 ) )
	# cv2.waitKey()
	_, sure_bg = cv2.threshold( sure_bg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
#	sure_bg = cv2.morphologyEx( sure_bg, cv2.MORPH_CLOSE, kernel )
#	sure_bg = cv2.dilate( sure_bg, kernel, iterations=3 )
	cv2.imshow( 'temp', sure_bg )
	cv2.waitKey()

# Test application
if __name__ == '__main__' :
	# Find the image files
	filename = sorted( glob.glob( '{}/Image_*.png'.format( sys.argv[1] ) ) )
	# Read the image files
	images = []
	for file in filename : images.append( cv2.imread( file, cv2.IMREAD_GRAYSCALE ) )
	# Convert the images list into a numpy array
	images = np.array( images )
	# Create the mask
	CreateMask( images )
