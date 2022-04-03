#include <cassert>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <orb/orb.h>

namespace orb
{

ORBDetectorDescriptor::ORBDetectorDescriptor()
{
    printf("ORBDetectorDescriptor is constructed\n"); fflush(stdout);
}

void ORBDetectorDescriptor::detectAndCompute( cv::InputArray _image, \
                                              cv::InputArray _mask, \
                                              std::vector<cv::KeyPoint>& _keypoints, \
                                              cv::OutputArray _descriptors, \
                                              bool _useProvidedKeypoints )
{
    printf("ORBDetectorDescriptor::detectAndCompute is called\n"); fflush(stdout);
    orb_opencv->detectAndCompute( _image, _mask, _keypoints, _descriptors, _useProvidedKeypoints );
}



} // namespace orb