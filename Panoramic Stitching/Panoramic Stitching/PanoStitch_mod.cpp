#include <opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include<vector>


using namespace std;
using namespace cv;
//using namespace cv::xfeatures2d;



int main()
{
	//setBreakOnError(true);
	//Mat img = imread("lena.png", 0);
	Mat img_L = imread("imgsfortest.jpg", cv::IMREAD_GRAYSCALE); //Has to be grey to do feature detection
	Mat img_R = imread("imgfortest2.jpg", cv::IMREAD_GRAYSCALE); //Has to be grey to do feature detection
	Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	detector->detectAndCompute(img_L, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img_R, noArray(), keypoints2, descriptors2);


	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
	// Since SIFT is a floating-point descriptor NORM_L2 is used

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch>> knn_matches; // raelly weird but youre not using std vector so i guess it makes sense...
	// Okay so check it
	// the Dmatch object knn_matches is producing an object with 4 data enteries, queryIndx, trainIndx, imIndx, and distance
	// in the code below we are comparing the trainIndx with the knnmatches 0 and 1 the matches in the set and looking to see 
	// which distance is closer etc. etc. 
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);


	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.5f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	//-- Draw matches
	Mat img_matches;
	drawMatches(img_L, keypoints1, img_R, keypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj; //Let's create a point2f obj that stores the x,y coordinates of a match for img 1
	std::vector<Point2f> scene; //Let's create a point2f scene that stores the x,y coordinates of matches for img 2

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		// Note that pushback is adding new index vals to the vector obj
		// takes the queryIdx and trainIdx of the two vals and 
		// amends them one by one writing the x,y of match 1 and x,y of match 2
		
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt); 
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC, 5); //ransacReprojThreshold is 4... Maximum allowed reprojection error to treat a point pair as an inlier
	cout << "H:\n" << H << endl;
	cout << "\n" << img_L.size << endl;
	Mat img_stitch;



	namedWindow("image", WINDOW_NORMAL);
	imshow("image", img_stitch);
	namedWindow("image2", WINDOW_NORMAL);
	imshow("image2", img_L);
	waitKey(0);
	return 0;
}