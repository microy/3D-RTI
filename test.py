#! /usr/bin/env python
# -*- coding:utf-8 -*-

# External dependencies
import glob
import math
import os
import sys
import cv2
import numpy as np
import Mask
import Mesh

# Read the RTI files (images + light positions)
def ReadRTIFiles( path ) :
	# Find the light position file
	filename = glob.glob( '{}/*.lp'.format( path ) )[0]
	# Read the light position file
	lights = []
	with open( filename, 'r' ) as file :
		# Read the first line (image number)
		file.readline()
		# Read each light position
		for line in file :
			lights.append( line.split()[1:4] )
	# Convert the light position list to a numpy array
	lights = np.array( lights, dtype=np.float )
	# Find the image files
	filename = sorted( glob.glob( '{}/Image_*.png'.format( path ) ) )
	# Read the image files
	images = []
	for file in filename :
		# Read the image
		images.append( cv2.imread( file, cv2.IMREAD_GRAYSCALE ) )
	# Convert the images list into a numpy array
	images = np.array( images )
	# Test if a mask image is present
	filename = '{}/mask.png'.format( path )
	if os.path.isfile( filename ) :
		# Read the mask image
		mask = cv2.imread( filename, cv2.IMREAD_GRAYSCALE )
		# Apply the mask to every image
		images[ :, mask == 0 ] = 0
	else : Mask.CreateMask( images )
	# The return the light positions and the images
	return lights, images

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
#		Ib = images[ :, :, x ]
#		n = np.dot( lights_inv, Ib[:] )
#		p = np.sqrt( (n ** 2).sum( axis=1 ) )
#		print( Ib.shape, n.shape, p.shape )
		for y in range( height ) :
			I = images[:, y, x]
			n = np.dot( lights_inv, I )
			p = math.sqrt( (n ** 2).sum() )
#			if y == 0 : print( '---', I.shape, n.shape, p )
			if p > 0 : n = n / p
			if n[2] == 0 : n[2] = 1
			legit = 1
			for i in range( len( images ) ) :
				legit *= images[i][y, x] >= 0
			if legit :
				normals[y, x] = n
				pgrads[y, x] = n[0] / n[2]
				qgrads[y, x] = n[1] / n[2]
			else :
				normals[y, x] = [0, 0, 1]
				pgrads[y, x] = 0
				qgrads[y, x] = 0
	# Return the normals
	return normals, pgrads, qgrads

# Compute the depth map
def GetDepthMap( pgrads,  qgrads) :
	l = 1.0
	mu = 1.0
	rows = pgrads.shape[0]
	cols = pgrads.shape[1]
	P = cv2.dft( pgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	Q = cv2.dft( qgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
	Z = np.zeros( (rows, cols, 2) )
	for i in range(rows) :
		for j in range(cols) :
			if i != 0 or j != 0 :
				u = math.sin( i * 2.0 * math.pi / rows )
				v = math.sin( j * 2.0 * math.pi / cols )
				uv = u ** 2 + v ** 2
				d = ( 1 + l ) * uv + mu * ( uv ** 2 )
				Z[i, j, 0] = ( u*P[i, j, 1] + v*Q[i, j, 1]) / d
				Z[i, j, 1] = (-u*P[i, j, 0] - v*Q[i, j, 0]) / d
	Z[0, 0, 0] = 0.0
	Z[0, 0, 1] = 0.0
	Z = cv2.dft( Z, flags = cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
	return Z

# Main application
if __name__ == '__main__' :
	print( 'Reading input data...' )
	lights, images = ReadRTIFiles( sys.argv[1] )
	print( 'Computing normal map...' )
	normals, pgrads, qgrads = GetNormalMap( lights, images )
	# Convert the normal map into an image
	normalmap_image = cv2.cvtColor( normals.astype( np.float32 ), cv2.COLOR_BGR2RGB )
	# Write the normal map
	cv2.imwrite( 'normalmap.png',  normalmap_image  * 255.99 )
#	cv2.imshow( 'normalmap.png',  normalmap_image )
#	cv2.waitKey()
	print( 'Computing depth map...' )
	z = GetDepthMap( pgrads, qgrads )
	# Triangulate and export the mesh to a PLY file
	Mesh.ExportMesh( images[0].shape[0], images[0].shape[1], z, 'mesh.ply' )
