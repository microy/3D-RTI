#! /usr/bin/env python
# -*- coding:utf-8 -*-

# External dependencies
import sys
import cv2
import numpy as np
import Data
import Mesh
import Stereo

# Main application
if __name__ == '__main__' :
	# Read input files
	print( 'Reading input data...' )
	lights, images = Data.ReadRTIFiles( sys.argv[1] )
	# Compute normal map
	print( 'Computing normal map...' )
	normals, pgrads, qgrads = Stereo.GetNormalMap( lights, images )
	# Convert the normal map into an image
	normalmap_image = cv2.cvtColor( normals.astype( np.float32 ), cv2.COLOR_BGR2RGB )
	# Write the normal map
	cv2.imwrite( 'normalmap.png',  normalmap_image  * 255.99 )
	# Compute the depth map
	print( 'Computing depth map...' )
	z = Stereo.GetDepthMap( pgrads, qgrads )
	# Triangulate the depth map, and export the mesh to a PLY file
	Mesh.ExportMesh( z, 'mesh.ply' )
