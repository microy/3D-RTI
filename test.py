# -*- coding:utf-8 -*-

#
# Calibrate the light
#

# External dependencies
import math
import cv2
import numpy as np

NUM_IMGS = 12
CALIBRATION = "Images/Chrome/chrome."
MODEL = "Images/Rock/rock."

def GetBoundingBox( mask ) :
    _, contours, _ = cv2.findContours( np.copy( mask ), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    return cv2.boundingRect( contours[0] )

def GetLightDirFromSphere( Image, boundingbox ) :
    THRESH = 254
    x, y, w, h = boundingbox
    radius = w / 2.0
    _, Binary = cv2.threshold( Image, THRESH, 255, cv2.THRESH_BINARY )
    SubImage = Binary[ y:y+h, x:x+w ]
    m = cv2.moments( SubImage )
    cx = int( m['m10'] / m['m00'] )
    cy = int( m['m01'] / m['m00'] )
    x = (cy - radius) / radius
    y = (cx - radius) / radius
    z = math.sqrt( 1.0 - pow(x, 2.0) - pow(y, 2.0) )
    return [ x, y, z ]

def GlobalHeights( Pgrads,  Qgrads) :
    l = 1.0
    mu = 1.0
    rows = Pgrads.shape[0]
    cols = Pgrads.shape[1]
    P = cv2.dft( Pgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
    Q = cv2.dft( Qgrads, flags = cv2.DFT_COMPLEX_OUTPUT )
    Z = np.zeros( (rows, cols, 2) )
    for i in range(rows) :
        for j in range(cols) :
            if i == 0 or j == 0 : continue
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

# Convert the surface to a triangular mesh
def SurfaceToMesh( height, width, Z ) :
	# Generate the X-Y grid
	X, Y = np.meshgrid( range(width), range(height) )
	# Create the vertices
	vertices = np.array( (X.flatten(), Y.flatten(), Z.flatten()) ).T
	# Get the size of the grid
	nb_lines = len( X )
	nb_cols = len( Y )
	# Array of vertex indices
	vindex = np.array( range( nb_lines * nb_cols ) ).reshape( nb_lines, nb_cols )
	# Find the diagonal that minimizes the Z difference
	left_diagonal = np.absolute( Z[1:,1:] - Z[:-1,:-1] ) > np.absolute( Z[1:,:-1] - Z[:-1,1:] )
	# Flatten the array
	left_diagonal = left_diagonal.flatten()
	# Double the values (1 square -> 2 triangles)
	left_diagonal = np.dstack( (left_diagonal, left_diagonal) ).flatten()
	# Initialize the right diagonal face array
	faces = np.empty( ( 2 * (nb_lines - 1) * (nb_cols - 1), 3 ), dtype=np.int )
	# Initialize the left diagonal face array
	left_faces = np.empty( ( 2 * (nb_lines - 1) * (nb_cols - 1), 3 ), dtype=np.int )
	#
	# Right diagonal
	#
	# Create lower triangle faces
	faces[ ::2, 0 ] = vindex[   : nb_lines - 1,   : nb_cols - 1 ].flatten()
	faces[ ::2, 1 ] = vindex[   : nb_lines - 1, 1 : nb_cols     ].flatten()
	faces[ ::2, 2 ] = vindex[ 1 : nb_lines    , 1 : nb_cols     ].flatten()
	# Create upper triangle faces
	faces[ 1::2, 0 ] = vindex[   : nb_lines - 1,   : nb_cols - 1 ].flatten()
	faces[ 1::2, 1 ] = vindex[ 1 : nb_lines    , 1 : nb_cols     ].flatten()
	faces[ 1::2, 2 ] = vindex[ 1 : nb_lines    ,   : nb_cols - 1 ].flatten()
	#
	# Left diagonal
	#
	# Create lower triangle faces
	left_faces[ ::2, 0 ] = vindex[   : nb_lines - 1,   : nb_cols - 1 ].flatten()
	left_faces[ ::2, 1 ] = vindex[   : nb_lines - 1, 1 : nb_cols     ].flatten()
	left_faces[ ::2, 2 ] = vindex[ 1 : nb_lines    ,   : nb_cols - 1 ].flatten()
	# Create upper triangle faces
	left_faces[ 1::2, 0 ] = vindex[   : nb_lines - 1, 1 : nb_cols     ].flatten()
	left_faces[ 1::2, 1 ] = vindex[ 1 : nb_lines    , 1 : nb_cols     ].flatten()
	left_faces[ 1::2, 2 ] = vindex[ 1 : nb_lines    ,   : nb_cols - 1 ].flatten()
	#
	# Merge left and right diagonal faces
	#
	faces[ left_diagonal ] = left_faces[ left_diagonal ]
	# Return the mesh
	return vertices, faces


#
# Write the surface to a Stanford PLY file
#
def WritePly( filename, vertices, faces ) :
	# Define the PLY file header
	header = '''ply
format binary_little_endian 1.0
element vertex {vertex_number}
property float x
property float y
property float z
element face {face_number}
property list int int vertex_indices
end_header\n'''.format( vertex_number = len( vertices ), face_number = len( faces ) )
	# Open the target PLY file
	with open( filename, 'wb' ) as ply_file :
		# Write the header
		ply_file.write( header.encode( 'UTF-8' ) )
		# Write the vertex data
		ply_file.write( st.pack( '3f' * len( vertices ), *vertices.flatten() ) )
		# Add the number of indices to every face
		faces = np.insert( faces, 0, 3, axis = 1 )
		# Write the face data
		ply_file.write( st.pack( '4i' * len( faces ), *faces.flatten() ) )



# Main application
if __name__ == '__main__' :

    # Calibrate the light
    calibImages = []
    modelImages = []
    Lights = np.empty( (NUM_IMGS, 3) )
    mask = cv2.imread( CALIBRATION + "mask.png", cv2.IMREAD_GRAYSCALE )
    ModelMask = cv2.imread( MODEL + "mask.png", cv2.IMREAD_GRAYSCALE )
    bb = GetBoundingBox( mask )
    for i in range( NUM_IMGS ) :
        Calib = cv2.imread( CALIBRATION + str(i) + ".png", cv2.IMREAD_GRAYSCALE )
        tmp = cv2.imread( MODEL + str(i) + ".png", cv2.IMREAD_GRAYSCALE )
        Model = cv2.bitwise_and( tmp, tmp, mask = ModelMask )
        Lights[i] = GetLightDirFromSphere(Calib, bb)
        calibImages.append( Calib )
        modelImages.append( Model )
    # Estimate the normals
    height = calibImages[0].shape[1]
    width = calibImages[0].shape[0]
    _, LightsInv = cv2.invert( Lights, flags = cv2.DECOMP_SVD )
    Normals = np.zeros( (width, height, 3) )
    Pgrads = np.zeros( (width, height) )
    Qgrads = np.zeros( (width, height) )
    for x in range( width ) :
        for y in range( height ) :
            I = np.empty( NUM_IMGS )
            for i in range( NUM_IMGS ) :
                I[i] = modelImages[i][x][y]
            n = np.dot( LightsInv, I )
            p = math.sqrt( (n ** 2).sum() )
            if p > 0 : n = n / p
            if n[2] == 0 : n[2] = 1
            legit = 1
            for i in range(NUM_IMGS) :
                legit *= modelImages[i][x][y] >= 0
            if legit :
                Normals[x][y] = n
                Pgrads[x][y] = n[0]/n[2]
                Qgrads[x][y] = n[1]/n[2]
            else :
                Normals[x][y] = [0, 0, 1]
                Pgrads[x][y] = 0
                Qgrads[x][y] = 0

    # View the normal map
#    cv2.imshow( "Normalmap", cv2.cvtColor( np.array( Normals, dtype=np.float32 ), cv2.COLOR_BGR2RGB ) )
#    cv2.waitKey()

    # Global integration of surface normals
    Z = GlobalHeights( Pgrads, Qgrads )

    output = ''
    for x in range( width ) :
        for y in range( height ) :
#            output += str(x) + ' ' + str(y) + ' ' + str(Z[x,y]) + '\n'
            output += 'v {} {} {}\n'.format( float(x), float(y), Z[x,y] )
#    with open( 'pytest.txt', 'w' ) as file :
    with open( 'pytest.obj', 'w' ) as file :
        file.write( output )

    vertices, faces = SurfaceToMesh( height, width, Z )
    WritePly( "test.obj", vertices, faces )
