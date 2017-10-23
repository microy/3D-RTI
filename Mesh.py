# -*- coding:utf-8 -*-

# External dependencies
import struct as st
import numpy as np

# Create a mesh from a regular grid, compute the normals, and export it to a PLY file
def ExportPly( filename, z, normals ) :
	# Get grid size
	height, width = z.shape[:2]
	# Generate the X-Y grid
	x, y = np.meshgrid( np.arange( width ), np.arange( height ) )
	# Flip the depth image to match 3D space
	z = np.flip( -z, 0 )
	# Create the vertices
	vertices = np.array( [ x.flatten(), y.flatten(), z.flatten() ] ).T
	# Array of vertex indices
	vindex = np.arange( height * width ).reshape( height, width )
	# Initialize the face array
	faces = np.empty( ( (height - 1) * (width - 1), 4 ), dtype=np.int )
	# Create rectangular faces
	faces[ :, 0 ] = vindex[   : height - 1,   : width - 1 ].flatten()
	faces[ :, 1 ] = vindex[   : height - 1, 1 : width     ].flatten()
	faces[ :, 2 ] = vindex[ 1 : height , 1 : width ].flatten()
	faces[ :, 3 ] = vindex[ 1 : height , : width - 1 ].flatten()
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
	full_vertices = np.hstack( ( vertices, normals.reshape( (width * height, 3) ) ) )
	# Add the number of indices to every face
	faces = np.insert( faces, 0, 4, axis = 1 )
	# Open the target PLY file
	with open( filename, 'wb' ) as ply_file :
		# Write the header
		ply_file.write( header.encode( 'UTF-8' ) )
		# Write the vertex data
		ply_file.write( st.pack( '6f' * len( vertices ), *full_vertices.flatten() ) )
		# Write the face data
		ply_file.write( st.pack( '5i' * len( faces ), *faces.flatten() ) )

# Write a VRML file
def ExportVrml( filename, z, normals ) :
	# Get grid size
	height, width = z.shape[:2]
	# Flip the image to match 3D space and flatten the array
	z = np.flip( -z, 0 ).flatten()
	# Reshape the normal array
	n = normals.reshape( (width * height, 3) )
	# Generate the U-V grid for the texture coordinates
	u, v = np.meshgrid( np.linspace( 0, 1, width ), np.linspace( 0, 1, height ) )
	t = np.array( [ u.flatten(), v.flatten() ] ).T
	# File Header
	vrml = '#VRML V2.0 utf8\n'
	# Begin description
	vrml += 'Shape {\n'
	# Texture filename
	vrml += '  appearance Appearance {\n'
	vrml += '    texture ImageTexture {\n'
	vrml += '      url "albedo.png"\n'
	vrml += '    }\n'
	vrml += '  }\n'
	# Elevation grid
	vrml += '  geometry ElevationGrid {\n'
	vrml += '    xDimension {}\n'.format( width )
	vrml += '    zDimension {}\n'.format( height )
	vrml += '    xSpacing 1\n'
	vrml += '    zSpacing 1\n'
	# Vertex coordinates
	vrml += '    height [\n'
	for i in range( len(z) ) :
		vrml += '           {}{}\n'.format( z[i], ',' if i < len(z)-1 else '' )
	vrml += '    ]\n'
	# # Vertex normals
	vrml += '    normalPerVertex TRUE\n'
	vrml += '    normal Normal {\n'
	vrml += '      vector [\n'
	for i in range( len(n) ) :
		vrml += '        {} {} {}{}\n'.format( *(n[i], ',' if i < len(n)-1 else '') )
	vrml += '      ]\n'
	vrml += '    }\n'
	# Texture coordinates
	vrml += '    texCoord TextureCoordinate {\n'
	vrml += '      point [\n'
	for i in range( len(t) ) :
		vrml += '        {} {}{}\n'.format( *(t[i], ',' if i < len(t)-1 else '') )
	vrml += '      ]\n'
	vrml += '    }\n'
	# End elevation grid
	vrml += '  }\n'
	# End description
	vrml += '}\n'
	# Open the target file
	with open( filename, 'wb' ) as vrml_file :
		# Write the VRML file
		vrml_file.write( vrml.encode( 'UTF-8' ) )

# Write a X3D file
def ExportX3d( filename, z, normals ) :
	# Get grid size
	height, width = z.shape[:2]
	# Flip the image to match 3D space and flatten the array
	z = np.flip( -z, 0 ).flatten()
	# Reshape the normal array
	n = normals.reshape( (width * height, 3) )
	# Generate the U-V grid for the texture coordinates
	u, v = np.meshgrid( np.linspace( 0, 1, width ), np.linspace( 0, 1, height ) )
	t = np.array( [ u.flatten(), v.flatten() ] ).T
	# File Header
	x3d = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" "http://www.web3d.org/specifications/x3d-3.3.dtd">
<X3D profile="Interchange" version="3.3" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.3.xsd">
'''
	# Begin description
	x3d += '<Scene>\n'
	x3d += '<Shape>\n'
	# Elevation grid
	x3d += '<ElevationGrid xDimension=\'{}\' zDimension=\'{}\' height=\'{}\'/>\n'.format( width, height, z.flatten().tolist() )
	# End description
	x3d += '</Shape>\n'
	x3d += '</Scene>\n'
	x3d += '</X3D>\n'
	# Open the target file
	with open( filename, 'wb' ) as x3d_file :
		# Write the X3D file
		x3d_file.write( x3d.encode( 'UTF-8' ) )
