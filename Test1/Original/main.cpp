#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>

using namespace cv;
using namespace std;

cv::Mat globalHeights(cv::Mat Pgrads, cv::Mat Qgrads) {

    cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));

    float lambda = 1.0f;
    float mu = 1.0f;

    cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);
    for (int i=0; i<Pgrads.rows; i++) {
        for (int j=0; j<Pgrads.cols; j++) {
            if (i != 0 || j != 0) {
                float u = sin((float)(i*2*CV_PI/Pgrads.rows));
                float v = sin((float)(j*2*CV_PI/Pgrads.cols));

                float uv = pow(u,2)+pow(v,2);
                float d = (1.0f + lambda)*uv + mu*pow(uv,2);
                Z.at<cv::Vec2f>(i, j)[0] = (u*P.at<cv::Vec2f>(i, j)[1] + v*Q.at<cv::Vec2f>(i, j)[1]) / d;
                Z.at<cv::Vec2f>(i, j)[1] = (-u*P.at<cv::Vec2f>(i, j)[0] - v*Q.at<cv::Vec2f>(i, j)[0]) / d;
    //            if( i>240 && i<245 && j>240 && j<245) {
    //                cout << Z.at<cv::Vec2f>(i, j) << endl << flush;
    //            }

            }
        }
    }

    /* setting unknown average height to zero */
    Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
    Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;

    cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

//    cout << Z.at<cv::Vec2f>(0, 0) << endl << flush;
//    cout << Z.at<float>(0, 0) << endl << flush;
    return Z;
}

cv::Vec3f getLightDirFromSphere(Mat Image, Rect boundingbox) {

    const int THRESH    = 254;
    const float radius  = boundingbox.width / 2.0f;

    Mat Binary;
    threshold(Image, Binary, THRESH, 255, CV_THRESH_BINARY);
    Mat SubImage(Binary, boundingbox);

    /* calculate center of pixels */
    Moments m = moments(SubImage, false);
    Point center(m.m10/m.m00, m.m01/m.m00);

    /* x,y are swapped here */
    float x = (center.y - radius) / radius;
    float y = (center.x - radius) / radius;
    float z = sqrt(1.0 - pow(x, 2.0) - pow(y, 2.0));

    return Vec3f(x, y, z);
}

cv::Rect getBoundingBox(cv::Mat Mask) {

    std::vector<std::vector<cv::Point> > v;
    cv::findContours(Mask.clone(), v, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    assert(v.size() > 0);
    return cv::boundingRect(v[0]);
}

int main(int argc, char *argv[]) {

    const int NUM_IMGS          = 12;
    const string CALIBRATION    = "../Images/Chrome/chrome.";
    const string MODEL          = "../Images/Rock/rock.";

    ostringstream result_log;

    vector<Mat> calibImages;
    vector<Mat> modelImages;
    Mat Lights(NUM_IMGS, 3, CV_32F);
    Mat Mask = imread(CALIBRATION + "mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat ModelMask = imread(MODEL + "mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    Rect bb = getBoundingBox(Mask);

    result_log << "Bounding box: " << bb << endl;

    for (int i = 0; i < NUM_IMGS; i++) {
        Mat Calib = imread(CALIBRATION + to_string(i) + ".png",
                           CV_LOAD_IMAGE_GRAYSCALE);
        Mat tmp = imread(MODEL + to_string(i) + ".png",
                           CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Model;
        tmp.copyTo(Model, ModelMask);
        Vec3f light = getLightDirFromSphere(Calib, bb);
        Lights.at<float>(i, 0) = light[0];
        Lights.at<float>(i, 1) = light[1];
        Lights.at<float>(i, 2) = light[2];
        calibImages.push_back(Calib);
        modelImages.push_back(Model);
    }

    result_log << "Lights:\n" << Lights << endl;

    const int height    = calibImages[0].rows;
    const int width     = calibImages[0].cols;

    result_log << "Height: " << height << endl;
    result_log << "Width: " << width << endl;

    /* light directions, surface normals, p,q gradients */
    cv::Mat LightsInv;
    cv::invert(Lights, LightsInv, cv::DECOMP_SVD);

    result_log << "LightsInv:\n" << LightsInv << endl;

    cv::Mat Normals(height, width, CV_32FC3, cv::Scalar::all(0));
    cv::Mat Pgrads(height, width, CV_32F, cv::Scalar::all(0));
    cv::Mat Qgrads(height, width, CV_32F, cv::Scalar::all(0));

    /* estimate surface normals and p,q gradients */
    for (int x=0; x<width; x++) {
        for (int y=0; y<height; y++) {
            Vec<float, NUM_IMGS> I;
            for (int i = 0; i < NUM_IMGS; i++) {
                I[i] = modelImages[i].at<uchar>(Point(x,y));
            }
            cv::Mat n = LightsInv * cv::Mat(I);
            float p = sqrt(cv::Mat(n).dot(n));

            // if( x>240 && x<245 && y>240 && y<245) {
            //     cout << x << "-" << y << " : " << n.reshape(3) << p << endl << flush;
            // }

            if (p > 0) { n = n/p; }
            if (n.at<float>(2,0) == 0) { n.at<float>(2,0) = 1.0; }
            int legit = 1;
            /* avoid spikes ad edges */
            for (int i = 0; i < NUM_IMGS; i++) {
                legit *= modelImages[i].at<uchar>(Point(x,y)) >= 0;
            }
            if (legit) {
                Normals.at<cv::Vec3f>(cv::Point(x,y)) = n;
                Pgrads.at<float>(cv::Point(x,y)) = n.at<float>(0,0)/n.at<float>(2,0);
                Qgrads.at<float>(cv::Point(x,y)) = n.at<float>(1,0)/n.at<float>(2,0);
            } else {
                cv::Vec3f nullvec(0.0f, 0.0f, 1.0f);
                Normals.at<cv::Vec3f>(cv::Point(x,y)) = nullvec;
                Pgrads.at<float>(cv::Point(x,y)) = 0.0f;
                Qgrads.at<float>(cv::Point(x,y)) = 0.0f;
            }

        }
    }

    result_log << "Pgrads:\n" << Pgrads << endl;
    cv::imwrite("pgrads.png", Pgrads * 255.99);
    cv::imshow("pgrads.png", Pgrads);
    cv::imwrite("qgrads.png", Qgrads * 255.99);
    cv::imshow("qgrads.png", Qgrads);

    cv::Mat Normalmap;
    cv::cvtColor(Normals, Normalmap, CV_BGR2RGB);
    cv::imwrite("normalmap.png", Normalmap * 255.99);
    cv::imshow("Normalmap", Normalmap);

    /* global integration of surface normals */
    cv::Mat Z = globalHeights(Pgrads, Qgrads);

    ofstream file( "original.obj" );
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            file << "v " << (float)x << " " << (float)y << " " << Z.at<float>(y,x) << endl;
        }
    }
    file.close();

    ofstream result_log_file( "result.log" );
    result_log_file << result_log.str();
    result_log_file.close();

    cv::waitKey();
    return 0;
}
