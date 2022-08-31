#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include<Eigen/Dense>
#include <Eigen/Core>


#include <iostream>
#include <iomanip>
#include <ctype.h>
#include <algorithm> 
#include <iterator> 
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<double> getQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    Eigen::Quaterniond r(cos(M_PI/4), 0, 0, sin(M_PI/4));

    Eigen::Quaterniond new_q = r * q * r.inverse();

    std::vector<double> v(4);
    v[0] = new_q.x();
    v[1] = new_q.y();
    v[2] = new_q.z();
    v[3] = new_q.w();

    return v;
}


void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int count = 0;
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVectorSem(vector<cv::Point2f> &v, vector<uchar> status)
{
    int count = 0;
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (!status[i])
            v[j++] = v[i];
    v.resize(j);
}


void filterSem(Mat &imgSem, vector<cv::Point2f> &points)
{
    vector<uchar> status(points.size());

    for( size_t i = 0; i < points.size(); i++ ) 
    {
        cv::Vec3b color = imgSem.at<cv::Vec3b>(cv::Point(points[i].x,points[i].y));
        bool cars = ((int)color[0] == 0 && (int)color[1] == 0 && (int)color[2] == 255);
        bool road = ((int)color[0] == 0 && (int)color[1] == 244 && (int)color[2] == 0);

        if(cars || road )
        {
            status[i] = 1;
        }
    }

    reduceVectorSem(points, status);
}

void featureTracking(Mat &img_1, Mat &img_2, vector<Point2f> &points1, vector<Point2f> &points2)
{
    vector<uchar> status;
    vector<float> err;					
    Size winSize=Size(21,21);																								
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    reduceVector(points1,status);
    reduceVector(points2,status);

}

void featureDetection(Mat &img_1, vector<Point2f> &points1)
{
    vector<KeyPoint> keypoints_1;
    // Ptr<FeatureDetector> detector = ORB::create();
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    // detector->detect(img_1,keypoints_1);
    KeyPoint::convert(keypoints_1, points1, vector<int>());
}

bool getFileContent(string fileName, vector<double> & vec)
{
    // Open the File
    ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        cerr << "Cannot open the File : "<<fileName<<endl;
        return false;
    }
    string str;
    // Read the next line from File untill it reaches the end.
    while (getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            vec.push_back(stod(str));
    }
    //Close The File
    in.close();
    return true;
}

void triangulation(const vector<Point2f> &prevPts, const vector<Point2f> &currPts, const Mat &currR, const Mat &currT,  vector<Point3d> &points) 
{

    Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);

    Mat T2 = (Mat_<float>(3, 4) <<
                currR.at<double>(0, 0), currR.at<double>(0, 1), currR.at<double>(0, 2), currT.at<double>(0, 0),
                currR.at<double>(1, 0), currR.at<double>(1, 1), currR.at<double>(1, 2), currT.at<double>(1, 0),
                currR.at<double>(2, 0), currR.at<double>(2, 1), currR.at<double>(2, 2), currT.at<double>(2, 0));
    
    Mat pts_4d;
    triangulatePoints(T1, T2, prevPts, currPts, pts_4d);

    for (int i = 0; i < pts_4d.cols; i++) 
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(2, 0);

        Point3d p( x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0) );
        points.push_back(p);
    }
}