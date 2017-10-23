# -*- coding:utf-8 -*-

#
# Estimate the normals from different images and light positions
#

# External dependencies
import cv2
import numpy as np

# Source adapted from https://github.com/NewProggie/Photometric-Stereo
def GetNormalMap1( lights, images ) :
	# Get the image size
	height, width = images[0].shape[ :2 ]
	# Compute the pseudo-inverse of the light position matrix using SVD
	_, lights_inv = cv2.invert( lights, flags = cv2.DECOMP_SVD )
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

# Source adapted from http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
def GetNormalMap2( lights, images ) :
	lights_t = lights.transpose()
	A = lights_t.dot( lights )
	A_inv = np.linalg.inv( A )
	# # v1
	# nrows, ncols = images[0].shape[:2]
	# normals = np.zeros( ( nrows, ncols, 3 ) )
	# albedo = np.zeros( ( nrows, ncols ) )
	# for i in range( nrows ) : # y
	# 	for j in range( ncols ) : # x
	# 		I = images[ :, i, j ]
	# 		b = lights_t.dot( I )
	# 		g = A_inv.dot( b )
	# 		R = np.sqrt( ( g ** 2 ).sum() )
	# 		N = g / R
	# 		if np.sqrt( ( I ** 2 ).sum() ) < 1.0E-06 :
	# 			N = 0
	# 			R = 0
	# 		normals[ i, j ] =  N
	# 		albedo[ i, j ] = R
	# v2
	# Get the image size
	height, width = images[0].shape[ :2 ]
	# Initialize the normals
	normals = np.zeros( ( height, width, 3 ) )
	albedo = np.zeros( ( height, width ) )
	for y in range( height ) :
		# Compute the normals
		b = np.dot( lights_t, images[ :, y, : ] )
		g = np.dot( A_inv, b ).T
		# Compute the albedo
		R = np.sqrt( ( g ** 2 ).sum( axis = 1 ) )
		# Normalize the normals
		valid = R > 0
		g[  valid ] /= R[ valid, np.newaxis ]
		g[ ~valid ]  = [ 0, 0, 1 ]
		# Save the normals and the albedo
		normals[ y, : ] =  g
		albedo[ y, : ] = R
	# # v3
	# # Compute the normals
	# normals = np.tensordot( lights_t, images, 1 )
	# normals = np.tensordot( A_inv, normals, 1 ).T.swapaxes( 0, 1 )
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
