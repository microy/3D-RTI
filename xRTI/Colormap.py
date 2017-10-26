# -*- coding:utf-8 -*-

# External dependency
import numpy as np

# Jet colormap
def Colormap( values ) :
	min, max = values.min(), values.max()
	vn = ( ( values - min ) / ( max - min ) ).flatten()
	c = np.zeros( ( len( vn ), 3 ) )
	for i, v in enumerate( vn ) :
		if v < 0.125 : c[i] = [ 0.0, 0.0, 0.5 + 0.5 * v * 8.0 ]
		elif v < 0.375 : c[i] = [ 0.0, (v - 0.125) * 4.0, 1.0 ]
		elif v < 0.625 : c[i] = [ 4.0 * ( v - 0.375 ), 1.0, 1.0 - ( v - 0.375 ) * 4.0 ]
		elif v < 0.875 : c[i] = [ 1.0, 1.0 - 4.0 * ( v - 0.625 ), 0.0 ]
		else : c[i] = [ 1.0 - 4.0 * ( v - 0.875 ), 0.0, 0.0 ]
	h, w = values.shape[:2]
	return c.reshape( ( h, w, 3 ) )
