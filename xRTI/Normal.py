# -*- coding:utf-8 -*-

#
# Estimate the normals from different images and light positions
#

# External dependencies
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
def GetNormalMap( lights, images ) :
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

# Write an image of the slopes
def SlopeImage( normals ) :
	# Compute the slopes
	pgrads = normals[ :, :, 0 ] / normals[ :, :, 2 ]
	qgrads = normals[ :, :, 1 ] / normals[ :, :, 2 ]
	# Compute the gradients of the normals
#	pgrads, qgrads, rgrads = np.gradient( normals )
	# Mixed the slopes
	s = pgrads + qgrads
#	s = np.sqrt( pgrads ** 2 + qgrads ** 2 + rgrads ** 2 )
#	s = np.sqrt( pgrads ** 2 + qgrads ** 2 )
	# Normalize
	min, max = s.min(), s.max()
	s -= min
	s /= (max-min)
	# Invert the values
	s *= -1
	s += 1
	# Return the slopes
	return s
