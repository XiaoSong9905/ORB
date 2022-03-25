#pragma once

#include <string>
#include <vector>
#include <opencv2/features2d.hpp>

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
        ORBDetectorDescriptor( /* TODO: add parameter as needed */ );

        /**
         * @brief Destroy the ORBDetectorDescriptor object
         * 
         */
        virtual ~ORBDetectorDescriptor();

        /**
         * @brief Detect keypoint and compute the descriptor
         * 
         * @note detect() and compute() internally call detectAndCompute() in cv::Feature2D
         *  when use the ORBDetectorDescriptor, we should call detectAndCompute() function directely
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
                                       std::vector<KeyPoint>& keypoints, \
                                       cv::OutputArray descriptors, \
                                       bool useProvidedKeypoints=false ) override;

        /**
         * @brief Getname of this extractor, used in some opencv funciton
         * 
         * @return std::string 
         */
        virtual std::string getDefaultName() const override;
    
    protected:
        // TODO add helper function here
};

};