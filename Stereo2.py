# -*- coding:utf-8 -*-

#
# Source adapted from http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
#

# External dependencies
import numpy as np

# Estimate the normals
def GetNormalMap( lights, images ) :
	lights_t = lights.transpose()
	A = lights_t.dot( lights )
	A_inv = np.linalg.inv( A )
	# v1
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
	# for i in range( nrows ) : # y
	# 	I = images[ :, i, : ]
	# 	b = np.dot( lights_t, I )
	# 	g = np.dot( A_inv, b ).T
	# 	R = np.sqrt( ( g ** 2 ).sum( axis = 1 ) )
	# 	condition = R > 0
	# 	g[condition] /= R[condition].reshape( (-1, 1) )
	# 	g[~condition] = [0, 0, 1]
	# 	normals[ i, : ] =  g
	# 	albedo[ i, : ] = R
	# v3
	# Compute the normals
	normals = np.tensordot( lights_t, images, 1 )
	normals = np.tensordot( A_inv, normals, 1 ).T.swapaxes( 0, 1 )
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
