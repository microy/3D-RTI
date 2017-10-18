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
	# Initialize the normals, pgrads, qgrads matrices
	normals = np.zeros( (height, width, 3) )
	pgrads = np.zeros( (height, width) )
	qgrads = np.zeros( (height, width) )
	# Compute the normal for each pixel
	for x in range( width ) :
		Ib = images[ :, :, x ]
		n = np.dot( lights_inv, Ib[:,:] ).T
		p = np.sqrt( (n ** 2).sum( axis = 1 ) )
		for i in range ( 3 ) :
			n[:,i] = np.where( p > 0, n[:,i] / p, n[:,i] )
		n[:,2] = np.where( n[:,2] == 0, 1, n[:,2] )
		normals[:,x] = n
		pgrads[:, x] = n[:,0] / n[:,2]
		qgrads[:, x] = n[:,1] / n[:,2]
	# Return the normals and the gradients
	return normals, pgrads, qgrads

# Compute the depth map
def GetDepthMap( pgrads,  qgrads ) :
	l = 1.0
	mu = 1.0
	height, width = pgrads.shape[:2]
	P = cv2.dft( pgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	Q = cv2.dft( qgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	Z = np.zeros( (height, width, 2) )
	uu, vv = np.meshgrid( np.arange(height)*2*np.pi/height, np.arange(width)*2*np.pi/width )
	uuvv = np.sin( uu ) ** 2 + np.sin( vv ) ** 2
#	dd = uuvv * 2 + uuvv ** 2
#	print(dd[200:200])
	for y in range(height) :
		for x in range(width) :
			if y != 0 or x != 0 :
				u = math.sin( y * 2.0 * math.pi / height )
				v = math.sin( x * 2.0 * math.pi / width )
				uv = u ** 2 + v ** 2
				d = ( 1 + l ) * uv + mu * ( uv ** 2 )
	#			if y==200 and x==200 : print(d)
				Z[y, x, 0] = ( u*P[y, x, 1] + v*Q[y, x, 1]) / d
				Z[y, x, 1] = (-u*P[y, x, 0] - v*Q[y, x, 0]) / d
	Z[0, 0, 0] = 0.0
	Z[0, 0, 1] = 0.0
	Z = cv2.dft( Z, flags = cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
	return Z
