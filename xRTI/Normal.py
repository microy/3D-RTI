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
#   https://github.com/soravux/pms
#   https://github.com/NewProggie/Photometric-Stereo
#   http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
#   https://github.com/RafaelMarinheiro/PhotometricStereo
#
def GetNormals( lights, images ) :
	# Ravel the images
	I = np.vstack( i.ravel() for i in images )
	# Compute the pseudo-inverse of the light position matrix using SVD
	lights_inv = np.linalg.pinv( lights )
	# Compute the normals
	normals = lights_inv.dot( I ).T
	# Compute the albedo
	albedo = np.linalg.norm( normals, axis = 1 )
	# Normalize the normals
	valid = albedo > 0
	normals[  valid ] /= albedo[ valid, np.newaxis ]
	normals[ ~valid ]  = [ 0, 0, 1 ]
	# Normalize the albedo
	albedo /= albedo.max()
	# Reshape the arrays
	height, width = images[ 0 ].shape[ :2 ]
	normals = normals.reshape( ( height, width, 3 ) )
	albedo = albedo.reshape( ( height, width ) )
	# Return the normals and the albedo
	return normals, albedo

# Compute the slopes
def GetSlopes( normals ) :
	# Compute the slopes
	dx = - normals[ :, :, 0 ] / normals[ :, :, 2 ]
	dy = - normals[ :, :, 1 ] / normals[ :, :, 2 ]
	# Return the slopes
	return dx, dy
