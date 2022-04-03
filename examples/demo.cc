//
// Demo on using our orb feature detector
// Reference 
// 1. https://github.com/sunzuolei/orb
// 2. https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html
// 
#include <cstdio>
#include <vector>
#include <memory> // shared_ptr
#include <opencv2/highgui.hpp> // cv::imshow
#include <opencv2/features2d.hpp> // cv::DescriptorMatcher
#include <opencv2/opencv.hpp> // cv::Mat

#include <orb/orb.h> // Our own ORB package

int main ( int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("Usage: ./demo PATH_TO_IMAGE1, PATH_TO_IMAGE2\n");
        exit(1);
    }

    cv::Mat image1 = cv::imread ( argv[1], cv::IMREAD_COLOR );
    cv::Mat image2 = cv::imread ( argv[2], cv::IMREAD_COLOR );

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    std::unique_ptr<cv::Feature2D> orb_feature = std::make_unique<cv::Feature2D>( orb::ORBDetectorDescriptor() );
    orb_feature->detectAndCompute( image1, cv::noArray(), keypoints1, descriptors1 );
    orb_feature->detectAndCompute( image2, cv::noArray(), keypoints2, descriptors2 );

    cv::Ptr<cv::BFMatcher> matcher;
    std::vector<std::vector<cv::DMatch>> matches;
    matcher->radiusMatch( descriptors1, descriptors2, matches, 0.21 );

    // Draw matches
    cv::Mat image_matches;
    cv::drawMatches( image1, keypoints1, image2, keypoints2, matches, image_matches, \
        cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), \
        std::vector<std::vector<char> >(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow("Good Matches", image_matches );
    cv::imwrite("match_image.png", image_matches);
    cv::waitKey();

    return 0;
}
