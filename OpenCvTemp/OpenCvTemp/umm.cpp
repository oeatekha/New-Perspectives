#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/core/affine.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"


using namespace std;
using namespace cv;
//using namespace cv::xfeatures2d;



int main()
{
	//setBreakOnError(true);
	cout << "Hello World!";
	Mat img = imread("lena.png", 0);
	Mat img_grey = imread("lena.png", cv::IMREAD_GRAYSCALE); //Has to be grey to do feature detection

	//auto detector = SiftFeatureDetector::create();S
	Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	Mat output;
	drawKeypoints(img, keypoints, output);
	imwrite("sift_result.jpg", output);





	namedWindow("image", WINDOW_NORMAL);
	imshow("image", output);
	waitKey(0);
	return 0;
}