#! /usr/bin/env python
# -*- coding:utf-8 -*-

# External dependencies
import sys
import cv2
import numpy as np
# Internal dependencies
import Data
import Mesh
import Stereo1
import Stereo2

# Read input files
print( 'Reading input data...' )
lights = Data.ReadLights( sys.argv[1] )
images = Data.ReadImages( sys.argv[1] )

# Compute normal map
print( 'Computing normal map...' )
# Stereo 1
normals, albedo = Stereo1.GetNormalMap( lights, images )
# Stereo 2
#normals2, albedo2 = Stereo2.GetNormalMap( lights, images )

# Convert the normal map into an image
normalmap_image = cv2.cvtColor( normals.astype( np.float32 ), cv2.COLOR_BGR2RGB )
# Dipslay the normal map and the albedo
cv2.imshow( 'Normal Map',  normalmap_image )
cv2.imshow( 'Albedo',  albedo )
cv2.waitKey()
cv2.destroyAllWindows()
# Save the normal map and the albedo
cv2.imwrite( 'normalmap.png',  normalmap_image  * 255.99 )
cv2.imwrite( 'albedo.png',  albedo  * 255.99 )

# Compute the depth map
print( 'Computing depth map...' )
z = Stereo1.GetDepthMap( normals )

# Triangulate the depth map, and export the mesh to a PLY file
print( 'Exporting mesh...' )
Mesh.ExportMesh( z, 'mesh.wrl' )
