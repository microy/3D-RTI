# -*- coding:utf-8 -*-

# External dependencies
import glob
import cv2
import numpy as np

# Read the RTI light positions file
def ReadLights( path ) :
	# Find the file
	filename = glob.glob( '{}/*.lp'.format( path ) )[0]
	# Read the file
	lights = []
	with open( filename, 'r' ) as light_file :
		# Read the first line (image number)
		light_file.readline()
		# Read each light position
		for line in light_file :
			lights.append( line.split()[1:4] )
	# Return the light positions into a numpy array
	return np.array( lights, np.float )

# Read the RTI image files
def ReadImages( path ) :
	# Find the PNG image files in the given path
	image_files = sorted( glob.glob( '{}/*.png'.format( path ) ) )
	# Find the mask image
	mask_file = [ f for f in image_files if f.endswith( 'mask.png' ) ][0]
	# Remove the mask image from the image list
	image_files.remove( mask_file )
	# Read the image files
	images = []
	for f in image_files :
		# Read the image
		images.append( cv2.imread( f, cv2.IMREAD_GRAYSCALE ) )
	# Convert the images list into a numpy array
	images = np.array( images )
	# Read the mask image
	mask = cv2.imread( mask_file, cv2.IMREAD_GRAYSCALE )
	mask = mask == 0
	print( mask.shape, mask.dtype )
	# Apply the mask to every image
	images[ :, mask ] = 0
	# Return the images
	return images, mask
