#include <string>
#include <vector>
#include <orb/orb.h>

namespace orb
{

ORBDetectorDescriptor::ORBDetectorDescriptor( /* TODO: add parameter as needed */  )
{

}

ORBDetectorDescriptor::~ORBDetectorDescriptor()
{

}

void OrbFeatureDetector::detectAndCompute( cv::InputArray image, \
                                           cv::InputArray mask, \
                                           std::vector<KeyPoint>& keypoints, \
                                           cv::OutputArray descriptors, \
                                           bool useProvidedKeypoints )
{

}

std::string ORBDetectorDescriptor::getDefaultName() const
{
    return (cv::FeatureDetector::getDefaultName() + ".ORBDetectorDescriptor");
}

// TODO: add helper function implementation here

} // namespace orb