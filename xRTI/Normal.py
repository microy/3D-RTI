# -*- coding:utf-8 -*-

#
# Estimate the normals from different images and light positions
#

# External dependencies
#import cv2
import numpy as np

#
# Reference :
#   Photometric Method for Determining Surface Orientation from Multiple Images
#   Robert J. Woodham, Optical Engineering, 19(1), 139-144, 1980
#
# Source adapted from :
#   https://github.com/NewProggie/Photometric-Stereo
#   http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
#
def GetNormalMap1( lights, images ) :
	# Get the image size
	height, width = images[0].shape[ :2 ]
	# Compute the pseudo-inverse of the light position matrix using SVD
#	_, lights_inv = cv2.invert( lights, flags = cv2.DECOMP_SVD )
	lights_inv = np.linalg.pinv( lights )
	# v1
	# Initialize the normals
	normals = np.zeros( ( height, width, 3 ) )
	albedo = np.zeros( ( height, width ) )
	# Compute the normal for each pixel
	for y in range( height ) :
		# Compute the normals
		n = np.dot( lights_inv, images[:, y, :] ).T
		# Compute the albedo
		p = np.sqrt( ( n ** 2 ).sum( axis = 1 ) )
		# Normalize the normals
		valid = p > 0
		n[  valid ] /= p[ valid, np.newaxis ]
		n[ ~valid ]  = [ 0, 0, 1 ]
		# Save the normals and the albedo
		normals[ y, : ] = n
		albedo[ y, : ] = p
	# # v2
	# # Compute the normals
	# normals = np.tensordot( lights_inv, images, 1 ).T.swapaxes( 0, 1 )
	# # Compute the albedo
	# albedo = np.sqrt( ( normals ** 2 ).sum( axis = 2 ) )
	# # Normalize the normals
	# valid = albedo > 0
	# normals[  valid ] /= albedo[ valid, np.newaxis ]
	# normals[ ~valid ]  = [ 0, 0, 1 ]
	# Normalize the albedo
	albedo /= albedo.max()
	# Return the normals
	return normals, albedo
