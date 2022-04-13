//
// Demo on using opencv feature detector
// Reference 
// 1. https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_16-02.cpp
// 2. https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html
// 
#include <cstdio>
#include <string>
#include <vector>
#include <memory> // shared_ptr
#include <opencv2/opencv.hpp> // cv::Mat
#include <opencv2/features2d.hpp>
#include <brisk/brisk.h>
#include <orb/orb.h> // Our own ORB package

void detection_and_matching(char** argv) {
    cv::Mat image1 = cv::imread ( argv[1], cv::IMREAD_COLOR );
    cv::Mat image2 = cv::imread ( argv[2], cv::IMREAD_COLOR );

    cv::Mat image1_grayscale;
    cv::Mat image2_grayscale;

    cv::cvtColor(image1, image1_grayscale, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, image2_grayscale, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    std::string type{ argv[3] };
    // detect and describe
    if (type == "orb_opencv") {
        printf("Build opencv ORB feature detector\n"); fflush( stdout );
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(image1, cv::noArray() , keypoints1, descriptors1);
        orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    } 
    else if (type == "brisk_opencv") {
        printf("Build opencv BRISK feature detector\n"); fflush( stdout );
        cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
        brisk->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
        brisk->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    } 
    else if (type == "orb") {
        printf("Build our own ORB feature detector\n"); fflush( stdout );
        orb::ORBDetectorDescriptor orb_feature{ 100, 0.5, 10, 5, 2 }; // NOTE: those value are just random value used to compile the program
        orb_feature.detectAndCompute( image1_grayscale, cv::noArray(), keypoints1, descriptors1 );
        orb_feature.detectAndCompute( image2_grayscale, cv::noArray(), keypoints2, descriptors2 );
    }
    else if (type == "brisk") {
        printf("Build BRISK feature detector\n"); fflush( stdout );
        float threshold = 34.0;     // "The keypoint detector threshold."
        int octaves = 4;
        bool suppressScaleNonmaxima = true;
        auto detector = new brisk::BriskFeatureDetector(threshold, octaves, suppressScaleNonmaxima);
        detector->detect(image1_grayscale, keypoints1);
        detector->detect(image2_grayscale, keypoints2);

        bool rotationInvariant = true;  // "If set to false, keypoints are assumed upright."
        bool scaleInvariant = true;     // "If set to false, keypoints are all assigned the same scale."
        auto descriptorExtractor = new brisk::BriskDescriptorExtractor(rotationInvariant, scaleInvariant, brisk::BriskDescriptorExtractor::Version::briskV2);
        descriptorExtractor->compute(image1_grayscale, keypoints1, descriptors1);
        descriptorExtractor->compute(image2_grayscale, keypoints2, descriptors2);
    }

    // Matching
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
    cv::imwrite("match_image.png", image_matches);
}

int main ( int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("Usage: ./demo PATH_TO_IMAGE1, PATH_TO_IMAGE2, DESCRIPTOR_TYPE\n");
        exit(1);
    }

    detection_and_matching(argv);

    return 0;
}