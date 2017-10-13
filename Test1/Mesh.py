# -*- coding:utf-8 -*-

# External dependencies
import struct as st
import numpy as np

# Create a mesh from a regular grid, compute the normals, and export it to a PLY file
def ExportMesh( height, width, Z, filename ) :
# Triangulate the grid
	# Generate the X-Y grid
	X, Y = np.meshgrid( np.arange( width ), np.arange( height ) )
	# Create the vertices
	vertices = np.array( [ X.flatten(), Y.flatten(), Z.flatten() ] ).T
	# Array of vertex indices
	vindex = np.arange( height * width ).reshape( height, width )
	# Initialize the right diagonal face array
	faces = np.empty( ( 2 * (height - 1) * (width - 1), 3 ), dtype=np.int )
	# Create lower triangle faces
	faces[ ::2, 0 ] = vindex[   : height - 1,   : width - 1 ].flatten()
	faces[ ::2, 1 ] = vindex[   : height - 1, 1 : width     ].flatten()
	faces[ ::2, 2 ] = vindex[ 1 : height    , 1 : width     ].flatten()
	# Create upper triangle faces
	faces[ 1::2, 0 ] = vindex[   : height - 1,   : width - 1 ].flatten()
	faces[ 1::2, 1 ] = vindex[ 1 : height    , 1 : width     ].flatten()
	faces[ 1::2, 2 ] = vindex[ 1 : height    ,   : width - 1 ].flatten()
	# Initialize the left diagonal face array
	left_faces = np.empty( ( 2 * (height - 1) * (width - 1), 3 ), dtype=np.int )
	# Create lower triangle faces
	left_faces[ ::2, 0 ] = vindex[   : height - 1,   : width - 1 ].flatten()
	left_faces[ ::2, 1 ] = vindex[   : height - 1, 1 : width     ].flatten()
	left_faces[ ::2, 2 ] = vindex[ 1 : height    ,   : width - 1 ].flatten()
	# Create upper triangle faces
	left_faces[ 1::2, 0 ] = vindex[   : height - 1, 1 : width     ].flatten()
	left_faces[ 1::2, 1 ] = vindex[ 1 : height    , 1 : width     ].flatten()
	left_faces[ 1::2, 2 ] = vindex[ 1 : height    ,   : width - 1 ].flatten()
	# Find the diagonal that minimizes the Z difference
	left_diagonal = np.absolute( Z[1:,1:] - Z[:-1,:-1] ) > np.absolute( Z[1:,:-1] - Z[:-1,1:] )
	# Flatten the array
	left_diagonal = left_diagonal.flatten()
	# Double the values (1 square -> 2 triangles)
	left_diagonal = np.dstack( (left_diagonal, left_diagonal) ).flatten()
	# Merge left and right diagonal faces
	faces[ left_diagonal ] = left_faces[ left_diagonal ]
# Compute the vertex normals
	# Create an indexed view of the triangles
	tris = vertices[ faces ]
	# Calculate the normal for all the triangles
	face_normals = np.cross( tris[::,1] - tris[::,0]  , tris[::,2] - tris[::,0] )
	# Normalise the face normals
	face_normals /= np.sqrt( ( face_normals ** 2 ).sum( axis=1 ) ).reshape( -1, 1 )
	# Initialise the vertex normals
	vertex_normals = np.zeros( vertices.shape )
	# Add the face normals to the vertex normals
	for i in range( 3 ) :
		vertex_normals[:, i] += np.bincount( faces[:, 0], face_normals[:, i], minlength=len( vertices ) )
		vertex_normals[:, i] += np.bincount( faces[:, 1], face_normals[:, i], minlength=len( vertices ) )
		vertex_normals[:, i] += np.bincount( faces[:, 2], face_normals[:, i], minlength=len( vertices ) )
	# Normalise the vertex normals
	vertex_normals /= np.sqrt( ( vertex_normals ** 2 ).sum( axis=1 ) ).reshape( -1, 1 )
# Write a PLY file
	# Define the PLY file header
	header = '''ply
format binary_little_endian 1.0
element vertex {vertex_number}
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face {face_number}
property list int int vertex_indices
end_header\n'''.format( vertex_number = len( vertices ), face_number = len( faces ) )
	# Merge vertices coordinates and normals
	full_vertices = np.hstack( ( vertices, vertex_normals ) )
	# Add the number of indices to every face
	faces = np.insert( faces, 0, 3, axis = 1 )
	# Open the target PLY file
	with open( filename, 'wb' ) as ply_file :
		# Write the header
		ply_file.write( header.encode( 'UTF-8' ) )
		# Write the vertex data
		ply_file.write( st.pack( '6f' * len( vertices ), *full_vertices.flatten() ) )
		# Write the face data
		ply_file.write( st.pack( '4i' * len( faces ), *faces.flatten() ) )
