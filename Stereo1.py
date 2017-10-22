# -*- coding:utf-8 -*-

#
# Source adapted from https://github.com/NewProggie/Photometric-Stereo
#

# External dependencies
import math
import cv2
import numpy as np

# Estimate the normals
def GetNormalMap( lights, images ) :
	# Get the image size
	height, width = images[0].shape[:2]
	# Compute the pseudo-inverse of the light position matrix using SVD
	_, lights_inv = cv2.invert( lights, flags = cv2.DECOMP_SVD )
	# Initialize the normals
	normals = np.zeros( (height, width, 3) )
	albedo = np.zeros( (height, width) )
	# Compute the normal for each pixel
	for y in range( height ) :
		I = images[:, y, :]
		n = np.dot( lights_inv, I ).T
		p = np.sqrt( ( n ** 2 ).sum( axis = 1 ) )
		condition = p > 0
		n[condition] /= p[condition].reshape( (-1, 1) )
		n[~condition] = [0, 0, 1]
		normals[y, :] = n
		albedo[y, :] = p
	# Normalize the albedo
	albedo /= albedo.max()
	# Return the normals
	return normals, albedo

# Compute the depth map
def GetDepthMap( normals ) :
	# Get the image size
	height, width = normals.shape[:2]
	# Initialize the pgrads, qgrads matrices
	pgrads = np.zeros( (height, width) )
	qgrads = np.zeros( (height, width) )
	# Compute the gradients
	for y in range( height ) :
		n = normals[y, :]
		pgrads[y, :] = n[:, 0] / n[:, 2]
		qgrads[y, :] = n[:, 1] / n[:, 2]
	# Compute the Fourier Transformation of the gradients
	P = cv2.dft( pgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	Q = cv2.dft( qgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	# Initilize the depth
	Z = np.zeros( (height, width, 2) )
#	z = np.zeros( (height, width, 2) )
	# v1
# 	for y in range(height) :
# 		for x in range(width) :
# 			if y == 0 and x == 0 : continue
# 			u = math.sin( y * 2.0 * math.pi / height )
# 			v = math.sin( x * 2.0 * math.pi / width )
# 			uv = u ** 2 + v ** 2
# 			d = 2 * uv + uv ** 2
# 			Z[y, x, 0] = ( u*P[y, x, 1] + v*Q[y, x, 1]) / d
# 			Z[y, x, 1] = (-u*P[y, x, 0] - v*Q[y, x, 0]) / d
	# v2
	u = np.sin( np.linspace( 0, 2 * np.pi, height, endpoint = False ) )
	v = np.sin( np.linspace( 0, 2 * np.pi, width, endpoint = False ) )
	uu, vv = np.meshgrid( np.linspace( 0, 2*np.pi, width, endpoint = False ), np.linspace( 0, 2*np.pi, height, endpoint = False ) )
	uv = np.sin( uu ) ** 2 + np.sin( vv ) ** 2
	d = 2 * uv + uv ** 2
	for y in range( height ) :
		Z[y, :, 0] = ( u[y]*P[y, :, 1] + v*Q[y, :, 1]) / d[y]
		Z[y, :, 1] = (-u[y]*P[y, :, 0] - v*Q[y, :, 0]) / d[y]
	Z[0, 0, 0] = 0.0
	Z[0, 0, 1] = 0.0
	# z[:, :, 0] = ( uu*P[:, :, 1] + vv*Q[:, :, 1]) / d
	# z[:, :, 1] = (-uu*P[:, :, 0] - vv*Q[:, :, 0]) / d
	# z[0, 0, 0] = 0.0
	# z[0, 0, 1] = 0.0
	# print( Z.shape, z.shape, np.allclose( Z, z ) )
	Z = cv2.dft( Z, flags = cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
	return Z
