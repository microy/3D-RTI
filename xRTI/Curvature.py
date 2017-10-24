# -*- coding:utf-8 -*-

#
# Kurita, T., & Boulanger, P. (1992). Computation of Surface Curvature from Range Images Using Geometrically Intrinsic Weights. In MVA (pp. 389-392).
# Zhao, C., Zhao, D., & Chen, Y. (1996, August). Simplified Gaussian and mean curvatures to range image segmentation. In Pattern Recognition, 1996., Proceedings of the 13th International Conference on (Vol. 2, pp. 427-431). IEEE.
#

# External dependency
import numpy as np

# Compute the mean and the gaussian curvature of a depth map
def GetCurvature( Z ) :
	# Compute the gradients
	Zy, Zx = np.gradient( Z )
	Zxy, Zxx = np.gradient( Zx )
	Zyy, _ = np.gradient( Zy )
	# Mean Curvature - equation (3) from Kurita and Boulanger (1992) paper
	# See also Surface in 3D space, http://en.wikipedia.org/wiki/Mean_curvature
	H = (1 + (Zx ** 2)) * Zyy + (1 + (Zy ** 2)) * Zxx - 2 * Zx * Zy * Zxy
	H = H / ((2 * (1 + (Zx ** 2) + (Zy ** 2))) ** 1.5)
	# Gaussian Curvature - equation (4) from Kurita and Boulanger (1992) paper
	K = (Zxx * Zyy - (Zxy ** 2)) /  ((1 + (Zx ** 2) + (Zy **2)) ** 2)
	# Simplified Mean Curvature - equation (3) from Zhao et.al (1996) paper
#	H = Zxx + Zyy
	#  Simplified Gaussian Curvature - equation (3) from Zhao et.al (1996) paper
#	K = Zxx * Zyy - (Zxy ** 2)
	# Return the mean and the gaussian curvature
	return H, K

# Jet colormap
def Colormap( values ) :
	min, max = values.min(), values.max()
	vn = ( ( values - min ) / ( max - min ) ).flatten()
	c = np.zeros( (len(vn), 3) )
	for i, v in enumerate( vn ) :
		if v < 0.125 : c[i] = [ 0.0, 0.0, 0.5 + 0.5 * v * 8.0 ]
		elif v < 0.375 : c[i] = [ 0.0, (v - 0.125) * 4.0, 1.0 ]
		elif v < 0.625 : c[i] = [ 4.0 * ( v - 0.375 ), 1.0, 1.0 - ( v - 0.375 ) * 4.0 ]
		elif v < 0.875 : c[i] = [ 1.0, 1.0 - 4.0 * ( v - 0.625 ), 0.0 ]
		else : c[i] = [ 1.0 - 4.0 * ( v - 0.875 ), 0.0, 0.0 ]
	h, w = values.shape[:2]
	return c.reshape( (h, w, 3) )
