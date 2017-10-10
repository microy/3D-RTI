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
    Normals = np.zeros( (height, width) )
    Pgrads = np.zeros( (height, width) )
    Qgrads = np.zeros( (height, width) )
    for x in range( width ) :
        for y in range( height ) :
            I = np.empty( NUM_IMGS )
            for i in range( NUM_IMGS ) :
                I[i] = modelImages[i][x][y]
                n = np.dot( LightsInv, I )
                p = np.sqrt( np.dot(n,np.transpose(n)) )
                if p > 0 : n = n / p
                
    #         cv::Mat n = LightsInv * cv::Mat(I);
    #         float p = sqrt(cv::Mat(n).dot(n));
    #         if (p > 0) { n = n/p; }
    #         if (n.at<float>(2,0) == 0) { n.at<float>(2,0) = 1.0; }
    #         int legit = 1;
    #         /* avoid spikes ad edges */
    #         for (int i = 0; i < NUM_IMGS; i++) {
    #             legit *= modelImages[i].at<uchar>(Point(x,y)) >= 0;
    #         }
    #         if (legit) {
    #             Normals.at<cv::Vec3f>(cv::Point(x,y)) = n;
    #             Pgrads.at<float>(cv::Point(x,y)) = n.at<float>(0,0)/n.at<float>(2,0);
    #             Qgrads.at<float>(cv::Point(x,y)) = n.at<float>(1,0)/n.at<float>(2,0);
    #         } else {
    #             cv::Vec3f nullvec(0.0f, 0.0f, 1.0f);
    #             Normals.at<cv::Vec3f>(cv::Point(x,y)) = nullvec;
    #             Pgrads.at<float>(cv::Point(x,y)) = 0.0f;
    #             Qgrads.at<float>(cv::Point(x,y)) = 0.0f;
    #         }
    #
    #     }
    # }
