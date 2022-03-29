#include <cassert>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <orb/orb.h>

namespace orb
{


void ORBDetectorDescriptor::detectAndCompute( cv::InputArray _image, \
                                           cv::InputArray _mask, \
                                           std::vector<cv::KeyPoint>& _keypoints, \
                                           cv::OutputArray _descriptors, \
                                           bool _useProvidedKeypoints )
{
    orb_opencv->detectAndCompute( _image, _mask, _keypoints, _descriptors, _useProvidedKeypoints );
}



} // namespace orb