# -*- coding:utf-8 -*-

#
# Estimate the normals from different images and light positions
#

# External dependencies
import numpy as np

#
# Compute the normals from a set of images and light directions
#
# Reference :
#   Photometric Method for Determining Surface Orientation from Multiple Images
#   Robert J. Woodham, Optical Engineering, 19(1), 139-144, 1980
#
# Source adapted from :
#   https://github.com/NewProggie/Photometric-Stereo
#   http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
#
def GetNormals( lights, images ) :
	# Get the image size
	height, width = images[0].shape[ :2 ]
	# Compute the pseudo-inverse of the light position matrix using SVD
	lights_inv = np.linalg.pinv( lights )
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
	# Normalize the albedo
	albedo /= albedo.max()
	# Return the normals
	return normals, albedo

# Compute the slopes
def GetSlopes( normals ) :
	# Compute the slopes
	dx = normals[ :, :, 0 ] / normals[ :, :, 2 ]
	dy = normals[ :, :, 1 ] / normals[ :, :, 2 ]
	# Compute the gradients of the normals
	dx, dy, _ = np.gradient( normals )
	# Return the slopes
	return dx, dy
