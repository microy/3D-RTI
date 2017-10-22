# -*- coding:utf-8 -*-

#
# Source adapted from https://github.com/NewProggie/Photometric-Stereo
#

# External dependencies
import cv2
import numpy as np

# Estimate the normals
def GetNormalMap( lights, images ) :
	# Get the image size
	height, width = images[0].shape[ :2 ]
	# Compute the pseudo-inverse of the light position matrix using SVD
	_, lights_inv = cv2.invert( lights, flags = cv2.DECOMP_SVD )
	# v1
	# Initialize the normals
	# normals = np.zeros( ( height, width, 3 ) )
	# albedo = np.zeros( ( height, width ) )
	# Compute the normal for each pixel
	# for y in range( height ) :
	# 	I = images[:, y, :]
	# 	n = np.dot( lights_inv, I ).T
	# 	p = np.sqrt( ( n ** 2 ).sum( axis = 1 ) )
	# 	condition = p > 0
	# 	n[condition] /= p[condition].reshape( (-1, 1) )
	# 	n[~condition] = [ 0, 0, 1 ]
	# 	normals[y, :] = n
	# 	albedo[y, :] = p
	# v2
	# Compute the normals
	normals = np.tensordot( lights_inv, images, 1 ).T.swapaxes( 0, 1 )
	# Compute the albedo
	albedo = np.sqrt( ( normals ** 2 ).sum( axis = 2 ) )
	# Normalize the normals
	valid = albedo > 0
	normals[  valid ] /= albedo[ valid, np.newaxis ]
	normals[ ~valid ]  = [ 0, 0, 1 ]
	# Normalize the albedo
	albedo /= albedo.max()
	# Return the normals
	return normals, albedo

# Compute the depth map
def GetDepthMap( normals ) :
	# Get the image size
	height, width = normals.shape[ :2 ]
	# Compute the gradients
	pgrads = normals[ :, :, 0 ] / normals[ :, :, 2 ]
	qgrads = normals[ :, :, 1 ] / normals[ :, :, 2 ]
	# Compute the Fourier Transformation of the gradients
	p = cv2.dft( pgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	q = cv2.dft( qgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	# Initialize the depth
	z = np.zeros( ( height, width, 2 ) )
	# v1
	# for y in range(height) :
	# 	for x in range(width) :
	# 		if y == 0 and x == 0 : continue
	# 		u = math.sin( y * 2.0 * math.pi / height )
	# 		v = math.sin( x * 2.0 * math.pi / width )
	# 		uv = u ** 2 + v ** 2
	# 		d = 2 * uv + uv ** 2
	# 		Z[y, x, 0] = ( u*P[y, x, 1] + v*Q[y, x, 1]) / d
	# 		Z[y, x, 1] = (-u*P[y, x, 0] - v*Q[y, x, 0]) / d
	# v2
	u = np.linspace( 0, 2 * np.pi, height, endpoint = False )
	v = np.linspace( 0, 2 * np.pi, width,  endpoint = False )
	u, v = np.meshgrid( np.sin( u ), np.sin( v ), indexing = 'ij' )
	uv = u ** 2 + v ** 2
	d = 2 * uv + uv ** 2
	# Fix division by zero
	d[ 0, 0 ] = 1
	# Compute the depth map
	z[ :, :, 0 ] = (  u * p[ :, :, 1 ] + v * q[ :, :, 1 ] ) / d
	z[ :, :, 1 ] = ( -u * p[ :, :, 0 ] - v * q[ :, :, 0 ] ) / d
	# Fix
	z[ 0, 0 ] = 0
	# Inverse Fourier transformation of the depth map
	z = cv2.dft( z, flags = cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
	# Return the depth map
	return z
