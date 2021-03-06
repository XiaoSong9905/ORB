//
// Demo on using our orb feature detector
// Reference 
// 1. https://github.com/sunzuolei/orb
// 2. https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html
// 
#include <cstdio>
#include <vector>
#include <memory> // shared_ptr
#include <opencv2/opencv.hpp> // cv::Mat
#include <orb/orb.h> // Our own ORB package

int main ( int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("Usage: ./demo PATH_TO_IMAGE1, PATH_TO_IMAGE2 PATH_TO_OUTPUT_IMAGE\n");
        exit(1);
    }

    cv::Mat image1 = cv::imread ( argv[1], cv::ImreadModes::IMREAD_GRAYSCALE );
    cv::Mat image2 = cv::imread ( argv[2], cv::ImreadModes::IMREAD_GRAYSCALE );

    if ( image1.empty() || image2.empty() )
    {
        printf("image file %s %s invalid\n", argv[1], argv[2] );
        exit(1);
    }

    // Resize image to 640, 480 to match intel realsense input
    cv::resize(image1, image1, cv::Size(640, 480), cv::INTER_LINEAR);
    cv::resize(image2, image2, cv::Size(640, 480), cv::INTER_LINEAR);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    printf("Build our own ORB feature detector\n"); fflush( stdout );

    int num_features = 1200;
    float pyramid_scale_factor = 1.2f;
    int pyramid_num_level = 8;
    int fast_default_threshold = 20;
    int fast_min_threshold  = 7;

    orb::ORBDetectorDescriptor orb_feature{ num_features, pyramid_scale_factor, pyramid_num_level, fast_default_threshold, fast_min_threshold };
    
    printf("Run detect and compute\n"); fflush( stdout );
    orb_feature.detectAndCompute( image1, cv::noArray(), keypoints1, descriptors1 );
    orb_feature.detectAndCompute( image2, cv::noArray(), keypoints2, descriptors2 );

    printf("Build opencv matcher\n"); fflush( stdout );
    cv::BFMatcher matcher;
    std::vector<std::vector<cv::DMatch>> matches;
    matcher.radiusMatch( descriptors1, descriptors2, matches, 0.21 );

    // Draw matches
    printf("Draw matching\n"); fflush( stdout );
    cv::Mat image_matches;
    cv::drawMatches( image1, keypoints1, image2, keypoints2, matches, image_matches, \
        cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), \
        std::vector<std::vector<char> >(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imwrite( argv[3], image_matches);

    return 0;
}
