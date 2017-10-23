# -*- coding:utf-8 -*-

#
# Compute the depth map from the normals
#

# External dependencies
import cv2
import numpy as np

# Source adapted from https://github.com/NewProggie/Photometric-Stereo
def GetDepthMap1( normals ) :
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

# Source adapted from http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
def GetDepthMap2( normals ) :
	nrows, ncols = normals.shape[:2]
	M = np.zeros( ( nrows, ncols, 2 ) )
	b = np.zeros( ( nrows, ncols ) )
	# for i in range( nrows ) : # y
	# 	for j in range( ncols ) : # x
	#
	#
	# z = np.linalg.solve( M, b )
	# z = z - z.min()
	# return z
