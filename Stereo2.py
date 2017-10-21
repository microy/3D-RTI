# -*- coding:utf-8 -*-

#
# Source adapted from http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
#

# External dependencies
import numpy as np

# Estimate the normals
def GetNormalMap( lights, images ) :
	nrows, ncols = images[0].shape[:2]
	normals = np.zeros( ( nrows, ncols, 3 ) )
	normals[ :, :, 2 ] = 1
	albedo = np.zeros( ( nrows, ncols ) )
	lights_t = lights.transpose()
	A = lights_t.dot( lights )
	A_inv = np.linalg.inv( A )

	# V1
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

	# V2
	for i in range( nrows ) : # y
		I = images[ :, i, : ]
		b = lights_t.dot( I )
		g = A_inv.dot( b ).T
		R = np.sqrt( ( g ** 2 ).sum( axis = 1 ) )
		N = g / R.reshape( (-1, 1) )
		normals[ i, : ] =  N
		albedo[ i, : ] = R

	maxval = albedo.max()
	if maxval > 0 : albedo /= maxval
	return normals, albedo


# Compute the depth map
def GetDepthMap( normals ) :
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
