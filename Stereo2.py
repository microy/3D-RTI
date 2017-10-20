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
	for i in range( nrows ) : # y
		for j in range( ncols ) : # x
			I = images[:, i, j]
			if i == 1000 and j == 1000: print( I.shape )
			I = I.transpose()
			if i == 1000 and j == 1000: print( I.shape )
			b = lights_t.dot( I )
			g = A_inv.dot( b )
			R = np.linalg.norm( g )
			N = g / R
			if np.linalg.norm( I ) < 1.0E-06 :
				N = 0
				R = 0
			normals[ i, j ] =  N
			albedo[ i, j ]= R

	# V2
	# for i in range( nrows ) : # y
	# 	I = images[:, i, :]
	# 	I = I.transpose()
	# 	b = lights_t.dot( I )
	# 	print( I.shape )
	# 	g = np.linalg.inv( A ).dot( b )
	# 	R = np.linalg.norm( g )
	# 	N = g / R
	# 	if np.linalg.norm( I ) < 1.0E-06 :
	# 		N = 0
	# 		R = 0
	#
	# 	normals[ i, : ] =  N
	# 	albedo[ i, : ]= R



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
