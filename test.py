#! /usr/bin/env python
# -*- coding:utf-8 -*-

# External dependencies
import sys
import cv2
import numpy as np
import Data
import Mesh
import Stereo1
import Stereo2

# Main application
if __name__ == '__main__' :

	# Read input files
	print( 'Reading input data...' )
	lights = Data.ReadLights( sys.argv[1] )
	images = Data.ReadImages( sys.argv[1] )

	# Compute normal map
	print( 'Computing normal map...' )
	# Stereo 1
#	normals, pgrads, qgrads = Stereo1.GetNormalMap( lights, images )
	# Stereo 2
	normals, albedo = Stereo2.GetNormalMap( lights, images )

	# Convert the normal map into an image
	normalmap_image = cv2.cvtColor( normals.astype( np.float32 ), cv2.COLOR_BGR2RGB )
	cv2.imshow( 'Normal Map',  normalmap_image )
	cv2.imshow( 'Albedo',  albedo )
	cv2.waitKey()
	# Write the normal map
	cv2.imwrite( 'normalmap.png',  normalmap_image  * 255.99 )
	cv2.imwrite( 'albedo.png',  albedo  * 255.99 )

	# Compute the depth map
#	print( 'Computing depth map...' )
#	z = Stereo.GetDepthMap( pgrads, qgrads )

	# Triangulate the depth map, and export the mesh to a PLY file
#	print( 'Exporting mesh...' )
#	Mesh.ExportMesh( z, 'mesh.ply' )
