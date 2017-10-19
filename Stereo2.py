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
	for i in range( nrows ) : # y
		for j in range( ncols ) : # x
			I = images[:, i, j]
			NP, R = PixelNormal( I, lights )
			normals[ i, j ] =  NP
			albedo[ i, j ]= R
	maxval = albedo.max()
	if maxval > 0 : albedo /= maxval
	return normals, albedo

def PixelNormal( I, L ) :
	I = I.transpose()
	LT = L.transpose()
	A = LT.dot( L )
	b = LT.dot( I )
	g = np.linalg.inv( A ).dot( b )
	R = np.linalg.norm( g )
	N = g / R
	if np.linalg.norm( I ) < 1.0E-06 :
		N = 0
		R = 0
	return N, R

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
