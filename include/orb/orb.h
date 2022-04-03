#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace orb
{

class ORBDetectorDescriptor : public cv::Feature2D
{
    public:
        /**
         * @brief The ORB constructor
         * 
         * @param param-name param-content
         */
        ORBDetectorDescriptor( );

        /**
         * @brief Destroy the ORBDetectorDescriptor object
         * 
         */
        virtual ~ORBDetectorDescriptor() = default;

        /**
         * @brief Detect keypoint and compute the descriptor
         * 
         * @note detect() and compute() internally call detectAndCompute() in cv::Feature2D
         * @note when use the ORBDetectorDescriptor, we should call detectAndCompute() function directely
         * 
         * @param image Image
         * @param mask Mask specifying where to look for keypoints (optional). 
         *   It must be a 8-bit integer matrix with non-zero values in the region of interest.
         *   In our ORBDetectorDescriptor, mask is currently not supported
         * @param keypoints Detected keypoints. Should be empty container when input
         * @param descriptors Computed descriptors. Should be empty when input
         * @param useProvidedKeypoints use provided keypoints to compute descriptor
         *   In our ORBDetectorDescriptor, this argument should always be false since we compute keypoint
         *   inside the detectAndCompute() function
         * 
         */
        virtual void detectAndCompute( cv::InputArray image, \
                                       cv::InputArray mask, \
                                       std::vector<cv::KeyPoint>& keypoints, \
                                       cv::OutputArray descriptors, \
                                       bool useProvidedKeypoints=false ) override;

    private:
        cv::Ptr<cv::ORB> orb_opencv = cv::ORB::create();
};

};