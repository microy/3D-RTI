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
images, mask = rti.ReadImages( sys.argv[1] )

# import timeit
# print( 'Testing...' )
# print( timeit.timeit( lambda : rti.GetNormalMap( lights, images ), number=10 ) )

# Compute the normals and the albedo
print( 'Computing normals and albedo...' )
normals, albedo = rti.GetNormals( lights, images )
normals[ mask ] = [ 0, 0, 1 ]
albedo[ mask ] = 0

# Save the normal map and the albedo
print( 'Saving normal map and albedo...' )
# Convert the normals to a normal map
normalmap_image = normals.copy()
normalmap_image[ :, 0 ] += 1
normalmap_image[ :, 0 ] /= 2
normalmap_image[ :, 1 ] += 1
normalmap_image[ :, 1 ] /= 2
normalmap_image = cv2.cvtColor( normalmap_image.astype( np.float32 ), cv2.COLOR_RGB2BGR )
normalmap_image[ mask ] = 0
cv2.imwrite( 'normalmap.png',  normalmap_image  * 255.99 )
cv2.imwrite( 'albedo.png',  albedo  * 255.99 )

# Compute the slopes
print( 'Computing slopes...' )
dx, dy = rti.GetSlopes( normals )
dx[ mask ] = 0
dy[ mask ] = 0

# Plot the slopes
# import matplotlib.pyplot as plt
# X, Y = np.meshgrid( np.arange( 0, images[0].shape[1] ), np.arange( 0, images[0].shape[0] ) )
# skip = ( slice( None, None, 5 ), slice( None, None, 5 ) )
# fig, ax = plt.subplots()
# im = ax.imshow( cv2.normalize( normals, None, 0, 1, cv2.NORM_MINMAX ) )
# ax.quiver( X[ skip ], Y[ skip ], dx[ skip ], dy[ skip ] )
# ax.set( aspect=1, title='Normals and slopes')
# fig.colorbar( im )
# plt.show()

# Save the slopes
print( 'Saving slopes...' )
cv2.imwrite( 'slope-x.png', cv2.normalize( dx, None, 0, 255, cv2.NORM_MINMAX ) )
cv2.imwrite( 'slope-y.png', cv2.normalize( dy, None, 0, 255, cv2.NORM_MINMAX ) )
s = np.dstack( ( dx, dy, np.ones( dx.shape ) ) )
s /= np.sqrt( 1 + dx ** 2 + dy ** 2 )[ :, :, np.newaxis ]
cv2.imwrite( 'slopes.png', cv2.cvtColor( s.astype( np.float32 ) * 255.99, cv2.COLOR_RGB2BGR ) )

# Compute the depth
print( 'Computing depth...' )
z = rti.GetDepth( normals )
z[ mask ] = 0
# Multiply the Z coordinates by the focal length to correct perspective
z *= float( args.f )

# Save the depth map
print( 'Saving depth map...' )
cv2.imwrite( 'depthmap.png', cv2.normalize( z, None, 0, 255, cv2.NORM_MINMAX ) )

# Compute the curvature
print( 'Computing curvature...' )
H, K = rti.GetCurvaturesFromSlopes( dx, dy )
cv2.imwrite( 'curvature-mean.png', cv2.normalize( H, None, 0, 255, cv2.NORM_MINMAX ) )
cv2.imwrite( 'curvature-gaussian.png', cv2.normalize( K, None, 0, 255, cv2.NORM_MINMAX ) )

# Triangulate the depth map, and export the mesh to a PLY file
print( 'Exporting mesh...' )
rti.ExportPly( 'mesh.ply', z, normals )
#rti.ExportX3d( 'mesh.x3d', z, normals )
#rti.ExportVrml( 'mesh.wrl', z, normals )
