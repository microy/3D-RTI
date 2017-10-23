#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# External dependencies
import sys
import cv2
import numpy as np
# Internal dependencies
import Curvature
import Data
import Mesh
import Normal
import Depth

# Read input files
print( 'Reading input data...' )
lights = Data.ReadLights( sys.argv[1] )
images = Data.ReadImages( sys.argv[1] )

# import timeit
# print( 'Testing...' )
# print( timeit.timeit( lambda:Normal.GetNormalMap1( lights, images ), number=1 ) )
# print( timeit.timeit( lambda:Normal.GetNormalMap2( lights, images ), number=1 ) )

# Compute normal map
print( 'Computing normal map...' )
# Stereo 1
#normals, albedo = Normal.GetNormalMap1( lights, images )
# Stereo 2
normals, albedo = Normal.GetNormalMap2( lights, images )
# Convert the normal map into an image
normalmap_image = cv2.cvtColor( normals.astype( np.float32 ), cv2.COLOR_BGR2RGB )
# Dipslay the normal map and the albedo
# cv2.imshow( 'Normal Map',  normalmap_image )
# cv2.imshow( 'Albedo',  albedo )
# cv2.waitKey()
# cv2.destroyAllWindows()
# Save the normal map and the albedo
cv2.imwrite( 'normalmap.png',  normalmap_image  * 255.99 )
cv2.imwrite( 'albedo.png',  albedo  * 255.99 )

# Compute the depth map
print( 'Computing depth map...' )
z = Depth.GetDepthMap1( normals )

# Compute the curvature
# print( 'Computing curvature...' )
# H, K = Curvature.GetCurvature( z )
# cv2.imshow( 'H', Curvature.Colormap(H) )
# cv2.imshow( 'K', Curvature.Colormap(K) )
# cv2.waitKey()
# cv2.destroyAllWindows()

# Triangulate the depth map, and export the mesh to a PLY file
print( 'Exporting mesh...' )
Mesh.ExportPly( 'mesh.ply', z, normals )
#Mesh.ExportX3d( 'mesh.x3d', z, normals )
#Mesh.ExportVrml( 'mesh.wrl', z, normals )
