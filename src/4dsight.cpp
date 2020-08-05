#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <opencv2/xfeatures2d.hpp>



//#define HOME
#define _SIFT
/*#ifdef HOME
	#define path_big "/home/fmd/Documents/repo/4d_map/src/StarMap.png"
	#define path_small "/home/fmd/Documents/repo/4d_map/src/Small_area_rotated.png"
#endif
*/

int main()
{
	std::string path_big, path_small;
	std::cout<<"Please enter the path of the StarMap image: "<<std::endl;
	std::cin>>path_big;
	std::cout<<std::endl;
	std::cout<<"Please enter the path of the cropped/rotated image: "<<std::endl;
	std::cin>>path_small;
	std::cout<<std::endl;

	//reading input images
	cv::Mat big_star, small_star, big_star_g, small_star_g;
	big_star = cv::imread(path_big, cv::IMREAD_COLOR);
	small_star = cv::imread(path_small, cv::IMREAD_COLOR);

	//convert images to grayscale
	cv::cvtColor(big_star, big_star_g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(small_star, small_star_g, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoint, keypoint_o;
    cv::Mat desc, desc_o;

    //SIFT
    //used sift features because they are rotation invariant which is needed to solve the problem in our case
    #ifdef _SIFT

    //detecting the sift features then computing their descriptors
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(0,3,0.01);
    sift->detect(small_star_g, keypoint);
    sift->detect(big_star_g, keypoint_o);
    sift->compute(small_star_g, keypoint, desc);
    sift->compute(big_star_g, keypoint_o, desc_o);

    #endif


    //some couts for debug purpose
    std::cout<<desc<<std::endl<<std::endl;
    std::cout<<"descriptor matrix size: "<<desc.size()<<std::endl;
    std::cout<<"small image size: "<<small_star_g.size()<<std::endl;
    std::cout<<"original image size: "<<big_star_g.size()<<std::endl;
    std::cout<<"keypoint vector size: "<<keypoint.size()<<std::endl;
    std::cout<<"first keypoint: "<<keypoint[0].pt<<std::endl; 

    //drawing keypoint detected on the cropped image
    cv::Mat image_keyp;
    cv::drawKeypoints(small_star_g, keypoint, image_keyp, (255,255,255));
    cv::imshow("Detected_Keypoints", image_keyp);
    cv::waitKey(0);

    cv::Mat matched_im;

    #ifdef _SIFT
    //find the correspondeces with the brute force matcher with using hamming distance between the descriptors
    std::vector<std::vector<cv::DMatch>> matches;
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create();
    bfm->knnMatch(desc, desc_o, matches, 2);
    std::cout<<"sift matches size "<<matches[0].size()<<std::endl;
    std::vector<cv::DMatch> good_matches;

    //get the better correspondence between the 2 correspondences for each from that came from knn match where k is two in our case
    //in addition there is a distance threshold to eliminate bad correspondences
	for(int i = 0; i < matches.size(); i++){
		if( (matches[i][0].distance <= matches[i][1].distance) && (matches[i][0].distance<100) )
			good_matches.push_back(matches[i][0]);
		else if( (matches[i][0].distance > matches[i][1].distance) && (matches[i][1].distance<100) )
			good_matches.push_back(matches[i][1]);

    }
    //draw correspondences between the cropped image and the original image
    cv::drawMatches(small_star_g, keypoint, big_star_g, keypoint_o, good_matches, matched_im, (255,255,255), (255,255,255), std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    for(int i = 0; i < good_matches.size(); i++){
    	std::cout<<"distance: "<<good_matches[i].distance<<std::endl;
    }


    std::vector<cv::Point2f> small_s;
    std::vector<cv::Point2f> big_s;
    //put the correspondence point set in seperate vectors which will be used in finding the homography matrix in between those images
    for( int i = 0; i < good_matches.size(); i++ ){
        small_s.push_back( keypoint[ good_matches[i].queryIdx ].pt );
        big_s.push_back( keypoint_o[good_matches[i].trainIdx ].pt );
    }

    cv::Mat H = cv::findHomography(small_s, big_s, cv::RANSAC);

    //we are curious about the corner points of the cropped image in the original image, therefore we are collecting them in a vector
    std::vector<cv::Point2f> crop_corners;
    cv::Point2f c1(0.0f,0.0f), c2(small_star_g.cols, 0.0f ), c3(small_star_g.cols, small_star_g.rows ), c4(0.0f, small_star_g.rows ); 
    crop_corners.push_back(c1);
    crop_corners.push_back(c2);
    crop_corners.push_back(c3);
    crop_corners.push_back(c4);
    std::vector<cv::Point2f> original_corners(4);

    //transforming the corner points of the cropped image to the original image with using homography matrix
	cv::perspectiveTransform( crop_corners, original_corners, H);
	//in order to show the corner points in the original image, drawing lines and stating the corner points seperately on the image where the correspondences were drawn
	//RANSAC is used in order to eliminate the outliers in the correspondences, outliers are still shown in the match image 
	cv::line( matched_im, original_corners[0] + cv::Point2f( small_star_g.cols, 0), original_corners[1] + cv::Point2f( small_star_g.cols, 0), cv::Scalar(0, 255, 0), 3 );
    cv::line( matched_im, original_corners[1] + cv::Point2f( small_star_g.cols, 0), original_corners[2] + cv::Point2f( small_star_g.cols, 0), cv::Scalar( 0, 255, 0), 3 );
    cv::line( matched_im, original_corners[2] + cv::Point2f( small_star_g.cols, 0), original_corners[3] + cv::Point2f( small_star_g.cols, 0), cv::Scalar( 0, 255, 0), 3 );
    cv::line( matched_im, original_corners[3] + cv::Point2f( small_star_g.cols, 0), original_corners[0] + cv::Point2f( small_star_g.cols, 0), cv::Scalar( 0, 255, 0), 3 );
    
    cv::circle(matched_im, original_corners[0] + cv::Point2f( small_star_g.cols, 0), 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(matched_im, original_corners[1] + cv::Point2f( small_star_g.cols, 0), 5, cv::Scalar(0, 0, 255), -1);
	cv::circle(matched_im, original_corners[2] + cv::Point2f( small_star_g.cols, 0), 5, cv::Scalar(0, 0, 255), -1);
	cv::circle(matched_im, original_corners[3] + cv::Point2f( small_star_g.cols, 0), 5, cv::Scalar(0, 0, 255), -1);



    #endif


    
    cv::imshow("Frame", matched_im);
    cv::waitKey(0);
    //cv::imwrite("result.png",matched_im);
	return 0;
}