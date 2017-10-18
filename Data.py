# -*- coding:utf-8 -*-

# External dependencies
import glob
import os
import cv2
import numpy as np
import Mask

# Read the RTI files (images + light positions)
def ReadRTIFiles( path ) :
	# Find the light position file
	filename = glob.glob( '{}/*.lp'.format( path ) )[0]
	# Read the light position file
	lights = []
	with open( filename, 'r' ) as file :
		# Read the first line (image number)
		file.readline()
		# Read each light position
		for line in file :
			lights.append( line.split()[1:4] )
	# Convert the light position list to a numpy array
	lights = np.array( lights, dtype=np.float )
	# Find the image files
	filename = sorted( glob.glob( '{}/Image_*.png'.format( path ) ) )
	# Read the image files
	images = []
	for file in filename :
		# Read the image
		images.append( cv2.imread( file, cv2.IMREAD_GRAYSCALE ) )
	# Convert the images list into a numpy array
	images = np.array( images )
	# Test if a mask image is present
	filename = '{}/mask.png'.format( path )
	if os.path.isfile( filename ) :
		# Read the mask image
		mask = cv2.imread( filename, cv2.IMREAD_GRAYSCALE )
		# Apply the mask to every image
		images[ :, mask == 0 ] = 0
	else : Mask.CreateMask( images )
	# The return the light positions and the images
	return lights, images
