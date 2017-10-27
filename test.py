#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# External dependencies
import argparse
import sys
import cv2
import numpy as np

# Internal dependency
import xRTI as rti

# Command line argument parser
parser = argparse.ArgumentParser( description='Process RTI images.', usage='%(prog)s [options] image_folder' )
parser.add_argument( 'image_folder', nargs='?', default=None, help='Input RTI image file folder' )
parser.add_argument( '-f', default=1, action='store', help='Camera focal length (default=1)' )
args = parser.parse_args()

# Read input files
print( 'Reading input data...' )
lights = rti.ReadLights( sys.argv[1] )
images = rti.ReadImages( sys.argv[1] )

# import timeit
# print( 'Testing...' )
# print( timeit.timeit( lambda : rti.GetNormalMap( lights, images ), number=10 ) )

# Compute the normals and the albedo
print( 'Computing normals and albedo...' )
normals, albedo = rti.GetNormals( lights, images )

# Save the normal map and the albedo
print( 'Saving normal map and albedo...' )
normalmap_image = cv2.cvtColor( normals.astype( np.float32 ), cv2.COLOR_RGB2BGR )
cv2.imwrite( 'normalmap.png',  normalmap_image  * 255.99 )
cv2.imwrite( 'albedo.png',  albedo  * 255.99 )

# Compute the slopes
print( 'Computing slopes...' )
dx, dy = rti.GetSlopes( normals )

# Save the slopes
print( 'Saving slopes...' )
c = cv2.normalize( dx, 0, 255, cv2.NORM_MINMAX ).astype( np.uint8 )
#c = cv2.applyColorMap( c, cv2.COLORMAP_RAINBOW )
cv2.imwrite( 'slope-x.png',  c )
c = cv2.normalize( dy, 0, 255, cv2.NORM_MINMAX ).astype( np.uint8 )
#c = cv2.applyColorMap( c, cv2.COLORMAP_RAINBOW )
cv2.imwrite( 'slope-y.png',  c )

# Compute the depth
print( 'Computing depth...' )
z = rti.GetDepth( normals )
# Multiply the Z coordinates by the focal length to correct perspective
z *= float( args.f )

# Save the depth map
print( 'Saving depth map...' )
zn = np.zeros( z.shape )
cv2.normalize( z, zn, 0, 255, cv2.NORM_MINMAX )
cv2.imwrite( 'depthmap.png', zn )

# Compute the curvature
# print( 'Computing curvature...' )
# H, K = rti.GetCurvature( z )
# cv2.imshow( 'H', Colormap(H) )
# cv2.imshow( 'K', Colormap(K) )
# cv2.waitKey()
# cv2.destroyAllWindows()

# Triangulate the depth map, and export the mesh to a PLY file
print( 'Exporting mesh...' )
rti.ExportPly( 'mesh.ply', z, normals )
#rti.ExportX3d( 'mesh.x3d', z, normals )
#rti.ExportVrml( 'mesh.wrl', z, normals )
