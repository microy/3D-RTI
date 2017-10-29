# -*- coding:utf-8 -*-

# External dependency
import numpy as np

#
# Compute the mean and the gaussian curvature of a depth map
#
# Reference :
#
#   Computation of Surface Curvature from Range Images Using Geometrically Intrinsic Weights
#   T. Kurita, P. Boulanger.
#   In MVA (pp. 389-392) 1992
#
def GetCurvaturesFromSlopes( Zx, Zy ) :
	# Compute the gradients
	Zxy, Zxx = np.gradient( Zx )
	Zyy, _ = np.gradient( Zy )
	# Mean Curvature
	H  = ( 1 + Zx ** 2 ) * Zyy + ( 1 + Zy ** 2 ) * Zxx - 2 * Zx * Zy * Zxy
	H /= ( 2 * ( 1 + Zx ** 2 + Zy ** 2 ) ) ** 1.5
	# Gaussian Curvature
	K  = Zxx * Zyy - Zxy ** 2
	K /= ( 1 + Zx ** 2 + Zy ** 2 ) ** 2
	# Return the different curvatures
	return H, K

def GetCurvatures( Z ) :
	# Compute the gradients
	Zy, Zx = np.gradient( Z )
	Zxy, Zxx = np.gradient( Zx )
	Zyy, _ = np.gradient( Zy )
	# Mean Curvature
	H  = ( 1 + Zx ** 2 ) * Zyy + ( 1 + Zy ** 2 ) * Zxx - 2 * Zx * Zy * Zxy
	H /= ( 2 * ( 1 + Zx ** 2 + Zy ** 2 ) ) ** 1.5
	# Gaussian Curvature
	K  = Zxx * Zyy - Zxy ** 2
	K /= ( 1 + Zx ** 2 + Zy ** 2 ) ** 2
	# Principle Curvatures
#	Pmin = H - np.sqrt( H ** 2 - K )
#	Pmax = H + np.sqrt( H ** 2 - K )
	# Return the different curvatures
	return H, K

#
# Compute the mean and the gaussian simplified curvature of a depth map
#
# Reference :
#
#   Simplified Gaussian and mean curvatures to range image segmentation
#   C. Zhao, D. Zhao, Y. Chen
#   In Proceedings of the 13th IEEE International Conference on Pattern Recognition (Vol. 2, pp. 427-431) 1996
#
def GetCurvaturesSimplified( Z ) :
	# Compute the gradients
	Zy, Zx = np.gradient( Z )
	Zxy, Zxx = np.gradient( Zx )
	Zyy, _ = np.gradient( Zy )
	# Simplified Mean Curvature
	H = Zxx + Zyy
	# Simplified Gaussian Curvature
	K = Zxx * Zyy - ( Zxy ** 2 )
	# Return the different curvatures
	return H, K
